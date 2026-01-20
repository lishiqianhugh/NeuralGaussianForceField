import sys
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
from utils.general_utils import setup_seed
from utils.transformation_utils import euler_xyz_to_matrix

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='NGFF Training Script')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--dynamic_model', type=str, default='ngff', choices=['ngff', 'pointformer', 'segno', 'gcn', 'sgnn'], help='Type of dynamic model to use')
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
    return parser.parse_args()

if __name__ == "__main__":
    #################################
    #       Workspace config        #
    #################################
    args = parse_args()
    from time import strftime
    t = strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"./exps/{args.dynamic_model}/out_{t}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_seed(args.seed)

    import logging
    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Training started at {t}")
    print(f"Training started at {t}")
    logging.info(f"Arguments: {args}")
    import json
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    dtype = torch.float32
    #################################
    #   Loading initial GS data     #
    #################################
    data_path = './data/GSCollision/mpm/'
    scenes = [os.path.join(data_path, group, scene) for group in os.listdir(data_path) if not group.startswith('.') for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
    scenes = [os.path.join(data_path, group, scene) for group in ['3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8'] if not group.startswith('.') for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
    scenes = [f'{data_path}/3_0/5_cloth_phone_bowl']
    scenes = scenes[:args.sample_num]
    t1 = time.time()
    dataset = DGSDataset(scenes, num_frames=args.steps, num_keypoints=args.num_keypoints, chunk=args.chunk, dtype=dtype, cache_dir='./data/GSCollision/cache')
    t2 = time.time()
    print(f"Dataset loaded in {t2-t1:.2f}s, total samples: {len(dataset)}")
    logging.info(f"Dataset loaded in {t2-t1:.2f}s, total samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  # batch of full sequences

    #################################
    #   Training NGFF simulation    #
    #################################
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dynamic_model == 'ngff':
        model = NGFFobj(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, mass=args.mass, 
                        dt=args.dt, ode_method=args.ode_method, r=0.1, step_size=args.step_size, threshold=args.threshold, rtol=args.rtol, atol=args.atol)
    elif args.dynamic_model == 'pointformer':
        model = PointTransformer(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
    elif args.dynamic_model == 'gcn':
        model = GCN(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, r=0.1)
    else:
        raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
    model.to(device, dtype=dtype)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 2e-4 1e-3

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr) # 5e-5 2e-4
    # copy train.py and model/ngff.py into output
    import shutil
    shutil.copy(__file__, os.path.join(save_dir, 'train.py'))
    shutil.copy(os.path.join(os.path.dirname(__file__), f'{args.dynamic_model}.py'), os.path.join(save_dir, f'{args.dynamic_model}.py'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'interaction_net.py'), os.path.join(save_dir, 'interaction_net.py'))
    
    # Training loop
    best_loss = 1e10
    losses = []
    for epoch in range(args.epochs):
        t2 = time.time()
        for i, data in enumerate(dataloader):
            t1 = time.time()
            # print(f"Data Time: {t1-t2:.6f}s")
            optimizer.zero_grad()
            points = data['points'].to(device)
            com = data['com'].to(device)
            com_vel = data['com_vel'].to(device)
            angle = data['angle'].to(device)
            angle_vel = data['angle_vel'].to(device)
            padding = data['padding'].to(device)
            knn = data['knn'].to(device) if 'knn' in data else None

            t2 = time.time()
            # print(f"Process Time: {t2-t1:.6f}s")
            # points [B, steps, num_objs, num_keypoints, 3], com [B, steps, num_objs, 3], com_vel [B, steps, num_objs, 3], angle [B, steps, num_objs, 3], angle_vel [B, steps, num_objs, 3], padding [B, num_objs, num_keypoints]
            # pred_points = model(points[:,0], com[:,0], com_vel[:,0], angle[:,0], angle_vel[:,0], padding, knn=knn, pred_len=args.chunk-1)
            if args.dynamic_model == 'ngff':
                point_seq, com_seq, angle_seq = model(points[:,0], com[:,0], com_vel[:,0],
                                                    angle[:,0], angle_vel[:,0], padding, knn,
                                                    pred_len=args.steps-1, external_forces=None)
                # compute point trajectory using translation, rotation, and deformation
                R_seq = euler_xyz_to_matrix(angle_seq)  # (B, T, num_objs, 3, 3)
                pred_points = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)  # (B, T, num_objs, N, 3)
            elif args.dynamic_model in ['pointformer', 'segno', 'gcn', 'sgnn'] :
                pred_points = model(points[:,0], com[:,0], com_vel[:,0],
                                angle[:,0], angle_vel[:,0], padding, knn,
                                pred_len=args.steps-1, external_forces=None)
            else:
                raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
            pred_points *= padding.unsqueeze(1).unsqueeze(-1)
            loss = F.mse_loss(pred_points, points) # pred_points: [B, steps, num_keypoints, 3], points: [B, steps, num_keypoints, 3]
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if epoch % args.save_interval == 0:
                # info = f"Epoch {epoch}, Loss: {loss.item():6f}, Pos Loss: {pos_loss.item():6f}, Com Loss: {com_loss.item():6f}"
                info = f"Epoch {epoch}, Loss: {loss.item():6f}"
                print(info)
                logging.info(info) # , feature_loss: {feature_loss.item():6f}
                torch.save(model.state_dict(), f"{save_dir}/ngff.pth")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), f"{save_dir}/ngff_best.pth")
                # draw log scale loss curves
                plt.plot(np.log(np.array(losses)))
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss curve')
                plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
                plt.close()
            t3 = time.time()
            # print(f"NN Time: {t3-t2:.6f}s")
    end_time = strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Training ended at {end_time}")
    print("Training complete. Saved in", save_dir) 
    logging.info(f"Training ended at {end_time}")
    logging.info(f"Training complete. Saved in {save_dir}")

