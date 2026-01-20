import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .DGSDataset import DGSDataset, LazyDGSDataset
from dynamic_models import GCN,PointTransformer,NGFFobj
from utils.general_utils import setup_seed
from utils.transformation_utils import euler_xyz_to_matrix


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='NGFF Training Script with DDP')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic_model', type=str, default='ngff', choices=['ngff', 'pointformer', 'segno', 'gcn', 'sgnn'], help='Type of dynamic model to use')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--num_keypoints', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8, help='Number of nearest neighbors for spring mass model')
    parser.add_argument('--sample_num', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=2001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dt', type=float, default=2e-2)
    parser.add_argument('--step_size', type=float, default=5e-3)
    parser.add_argument('--mass', type=float, default=1e-1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--ode_method', type=str, default='euler', choices=['adaptive', 'euler'])
    parser.add_argument('--reload', type=str, default=None, help='Path to reload model')
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize distributed training
    torch.distributed.init_process_group("nccl")
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    print(f"Rank {rank}/{world_size} initializing...")
    setup_seed(args.seed + rank)
    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    # Directory setup only on rank 0
    if rank == 0:
        from time import strftime
        t = strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = f"./exps/{args.dynamic_model}/out_{t}"
        os.makedirs(save_dir, exist_ok=True)
        import logging
        logging.basicConfig(
            filename=os.path.join(save_dir, 'train.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"Training started at {t}")
        logging.info(f"Arguments: {args}")
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        import shutil
        shutil.copy(__file__, os.path.join(save_dir, 'train_ddp.py'))
        shutil.copy(os.path.join(os.path.dirname(__file__), 'ngff.py'), os.path.join(save_dir, 'ngff.py'))
        shutil.copy(os.path.join(os.path.dirname(__file__), 'interaction_net.py'), os.path.join(save_dir, 'interaction_net.py'))
    else:
        save_dir = None

    # Model setup
    if args.dynamic_model == 'ngff':
        model = NGFFobj(
            input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            num_layers=args.num_layers, num_keypoints=args.num_keypoints, k=args.k, mass=args.mass, dt=args.dt,
            ode_method=args.ode_method, r=0.1, step_size=args.step_size,
            threshold=args.threshold, rtol=args.rtol, atol=args.atol
        ).to(device, dtype=dtype)
    elif args.dynamic_model == 'pointformer':
        model = PointTransformer(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers).to(device, dtype=dtype)
    elif args.dynamic_model == 'gcn':
        model = GCN(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, r=0.1).to(device, dtype=dtype)
    else:
        raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
    if args.reload is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(args.reload, map_location=map_location))
        print(f"Model reloaded from {args.reload}")
        if rank == 0:
            logging.info(f"Model reloaded from {args.reload}")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=True)
    
    # Load dataset
    data_path = './data/GSCollision/mpm'
    scenes = [os.path.join(data_path, group, scene) for group in ['3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8'] for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
    scenes = scenes[:args.sample_num]
    t1 = time.time()
    dataset = DGSDataset(scenes, num_frames=args.steps, num_keypoints=args.num_keypoints, k=args.k, chunk=args.chunk, dtype=dtype, cache_dir='./data/GSCollision/cache')
    t2 = time.time()
    print(f"Dataset loaded in {t2-t1:.2f}s, total samples: {len(dataset)}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_loss = 1e10
    losses = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            points = data['points'].to(device)
            com = data['com'].to(device)
            com_vel = data['com_vel'].to(device)
            angle = data['angle'].to(device)
            angle_vel = data['angle_vel'].to(device)
            padding = data['padding'].to(device)
            knn = data['knn'].to(device)
            if args.dynamic_model == 'ngff':
                local_points_pred, com_seq, angle_seq = model(
                    points[:, 0], com[:, 0], com_vel[:, 0],
                    angle[:, 0], angle_vel[:, 0], padding, knn,
                    pred_len=args.steps - 1
                )
                pred_points = torch.matmul(local_points_pred, euler_xyz_to_matrix(angle_seq).transpose(-2, -1)) + com_seq.unsqueeze(-2)
            elif args.dynamic_model in ['pointformer', 'segno', 'gcn', 'sgnn'] :
                pred_points = model(points[:,0], com[:,0], com_vel[:,0],
                                angle[:,0], angle_vel[:,0], padding, knn,
                                pred_len=args.steps-1, external_forces=None)
            else:
                raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
            # pred_points = local_points_pred + com_seq.unsqueeze(-2)  # (B, N, 3)
            pred_points *= padding.unsqueeze(1).unsqueeze(-1)

            pos_loss = torch.mean((pred_points - points)**2)
            loss = pos_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        if rank == 0 and epoch % args.save_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            logging.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            torch.save(model.module.state_dict(), f"{save_dir}/ngff.pth")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.module.state_dict(), f"{save_dir}/ngff_best.pth")
            plt.plot(np.log(np.array(losses)))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss curve')
            plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
            plt.close()

    if rank == 0:
        end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        print(f"Training complete. Saved in {save_dir}. Ended at {end_time}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()