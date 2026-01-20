import warnings
warnings.filterwarnings("ignore")
import sys
import shutil
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import time
import os
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .DGSDataset import DGSDataset, LazyDGSDataset
from dynamic_models import GCN,PointTransformer,NGFFobj
from accelerate import Accelerator
from tqdm import tqdm
import logging
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs,ProfileKwargs

from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from utils.transformation_utils import euler_xyz_to_matrix

logger = get_logger(__name__, log_level="INFO")

def configure_logging(save_dir: str, accel_logger):
    """Attach file and console handlers to the provided accelerate logger.
    """
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, 'train.log')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    accel_logger.logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    accel_logger.logger.addHandler(ch)

    accel_logger.logger.setLevel(logging.INFO)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='NGFF Training Script')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--dynamic_model', type=str, default='ngff', choices=['ngff', 'pointformer', 'gcn'], help='Type of dynamic model to use')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps in the sequence')
    parser.add_argument('--chunk', type=int, default=100, help='Chunk size for training')
    parser.add_argument('--num_keypoints', type=int, default=4096, help='Number of keypoints for a scene')
    parser.add_argument('--sample_num', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=2001, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--dt', type=float, default=2e-2, help='Time interval of observed frames')
    parser.add_argument('--step_size', type=float, default=5e-3, help='Step size for ODE integration')
    parser.add_argument('--mass', type=float, default=1e-1, help='Mass of the object points')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for the model')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension of the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--threshold', type=float, default=1e-1, help='Distance threshold for contact detection')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--min_lr', type=float, default=5e-5, help='Minimum learning rate for the scheduler')
    parser.add_argument('--save_interval', type=int, default=100, help='Interval for saving the model')
    parser.add_argument('--rtol', type=float, default=1e-3, help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance for ODE solver')
    parser.add_argument('--ode_method', type=str, default='euler', choices=['adaptive', 'euler'], help='ODE solver method')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'bf16', 'fp16'], help='Mixed precision training')

    # Optional logger
    parser.add_argument("--training_tracker", type=str, default=None, choices=["swanlab"], help="Logger to use (swanlab or None)")
    parser.add_argument("--project_name", type=str, default="NGFF-Dynamics", help="SwanLab project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="SwanLab experiment name")

    return parser.parse_args()

def prepare_dataset(args, dtype):
    #################################
    #   Loading initial GS data     #
    #################################
    data_path = './data/GSCollision/GSCollision/mpm/'
    # scenes = [os.path.join(data_path, group, scene) for group in os.listdir(data_path) if not group.startswith('.') for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
    scenes = [os.path.join(data_path, group, scene) for group in ['3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8'] if not group.startswith('.') for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
    scenes = scenes[:args.sample_num]

    t1 = time.time()
    dataset = DGSDataset(scenes, num_frames=args.steps, num_keypoints=args.num_keypoints, chunk=args.chunk, dtype=dtype, cache_dir='./data/GSCollision/cache')
    t2 = time.time()

    logger.info(f"Dataset loaded in {t2-t1:.2f}s, total samples: {len(dataset)}")
    
    return dataset

def prepare_model(args):
    if args.dynamic_model == 'ngff':
        model = NGFFobj(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, mass=args.mass, 
                        dt=args.dt, ode_method=args.ode_method, r=0.1, step_size=args.step_size, threshold=args.threshold, rtol=args.rtol, atol=args.atol)
    elif args.dynamic_model == 'pointformer':
        model = PointTransformer(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
    elif args.dynamic_model == 'gcn':
        model = GCN(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, r=0.1)
    else:
        raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
    
    return model

def configure_training_tracker(args, accelerator, time_str):
    config = {
        "dynamic_model": args.dynamic_model,
        "steps": args.steps,
        "chunk": args.chunk,
        "num_keypoints": args.num_keypoints,
        "sample_num": args.sample_num,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dt": args.dt,
        "step_size": args.step_size,
        "mass": args.mass,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "num_layers": args.num_layers,
        "threshold": args.threshold,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "save_interval": args.save_interval,
        "rtol": args.rtol,
        "atol": args.atol,
        "ode_method": args.ode_method,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
    }
        
    # Generate experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"{args.dynamic_model}_out_{time_str}"
    
    accelerator.init_trackers(
        project_name=args.project_name,
        config=config,
        init_kwargs={"swanlab": {"experiment_name": experiment_name}}
    )

def main():
    args = parse_args()

    from time import strftime
    time_str = strftime("%Y-%m-%d-%H-%M-%S")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator_kwargs = {
        "project_dir": f"./exps/{args.dynamic_model}/out_{time_str}",
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "kwargs_handlers": [ddp_kwargs],
    }

    if args.training_tracker == "swanlab":
        accelerator_kwargs["log_with"] = ["swanlab"]

    accelerator = Accelerator(**accelerator_kwargs)

    set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(accelerator.project_dir, exist_ok=True)
        with open(os.path.join(accelerator.project_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        # configure logging to use accelerate's logger and write to project dir
        configure_logging(accelerator.project_dir, accel_logger=logger)
        configure_training_tracker(args, accelerator, time_str)

    logger.info(f"Training started at {time_str}")
    logger.info(f"Arguments: {args}")

    dataset = prepare_dataset(args,dtype=torch.float32)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )

    model = prepare_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(dataloader)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.02 * num_training_steps, 
        num_training_steps=num_training_steps,
        min_lr=args.min_lr
    )

    if accelerator.is_main_process:
        shutil.copy(__file__, os.path.join(accelerator.project_dir, 'train.py'))
        shutil.copy(os.path.join(os.path.dirname(__file__), f'{args.dynamic_model}.py'), os.path.join(accelerator.project_dir, f'{args.dynamic_model}.py'))
        shutil.copy(os.path.join(os.path.dirname(__file__), 'interaction_net.py'), os.path.join(accelerator.project_dir, 'interaction_net.py'))

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    accelerator.wait_for_everyone()

    best_loss = 1e10

    progress_bar = tqdm(
        range(args.epochs),
        initial=0,
        desc="Epochs",
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
        ncols=100,
    )

    global_step = 0
    for epoch in progress_bar:
        with accelerator.accumulate(model):
            for i, data in enumerate(dataloader):
                optimizer.zero_grad()
                points = data['points']
                com = data['com']
                com_vel = data['com_vel']
                angle = data['angle']
                angle_vel = data['angle_vel']
                padding = data['padding']
                knn = data['knn'] if 'knn' in data else None

                if args.dynamic_model == 'ngff':
                    point_seq, com_seq, angle_seq = model(points[:,0], com[:,0], com_vel[:,0],
                                                        angle[:,0], angle_vel[:,0], padding, knn,
                                                        pred_len=args.steps-1, external_forces=None)
                    # compute point trajectory using translation, rotation, and deformation
                    R_seq = euler_xyz_to_matrix(angle_seq)  # (B, T, num_objs, 3, 3)
                    pred_points = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)  # (B, T, num_objs, N, 3)
                elif args.dynamic_model in ['pointformer', 'gcn'] :
                    pred_points = model(points[:,0], com[:,0], com_vel[:,0],
                                    angle[:,0], angle_vel[:,0], padding, knn,
                                    pred_len=args.steps-1, external_forces=None)
                else:
                    raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
                
                pred_points *= padding.unsqueeze(1).unsqueeze(-1)
                loss = F.mse_loss(pred_points, points) # pred_points: [B, steps, num_keypoints, 3], points: [B, steps,num_keypoints, 3]
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                # log scalars every 10 steps to avoid frequent GPU->CPU sync
                if args.training_tracker == "swanlab" and (global_step % 2 == 0):
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    },step=global_step)
                global_step += 1

            if epoch % args.save_interval == 0 and epoch != 0:
                info = f"Epoch {epoch}, Loss: {loss.item():6f}"
                logger.info(info)
                if accelerator.is_main_process:
                    torch.save(model.state_dict(), f"{accelerator.project_dir}/ngff.pth")
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(model.state_dict(), f"{accelerator.project_dir}/ngff_best.pth")
                
    progress_bar.close()

    end_time = strftime("%Y-%m-%d-%H-%M-%S")
    logger.info(f"Training ended at {end_time}")
    logger.info(f"Training complete. Saved in {accelerator.project_dir}")


if __name__ == "__main__":
    main()