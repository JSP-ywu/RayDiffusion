import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import configargparse

from ray_diffusion.dataset.co3d_v2 import Co3dDataset
from ray_diffusion.dataset.custom import CustomDataset

from ray_diffusion.utils.rays import cameras_to_rays

#from ray_diffusion.model.diffuser import RayDiffuser
#from ray_diffusion.model.dit import Dit
#from ray_diffusion.model.feature_extractors import SpatialDino
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras

# Custom Key for training configuration
CUSTOM_KEY = None

import wandb
with open('./wandb_key.txt', 'r') as f:
    wandb_key = f.readline()
wandb.login(key=wandb_key)
wandb.init(project="RayDiffusion - compute_x0")

'''
Parsing Argument
'''
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-dp", "--datapath", help="dataset path",
                        default='/data1')
    parser.add_argument('-op', '--outputpath', help="output path",
                        default='/data1')
    parser.add_argument("-multigpu", action='store_ture', help="whether use multiple gpu or not",
                        default=True)
    parser.add_argument("-ds", "--dataset", help="choose dataset <7scenes, Cambridge, Co3d_v2",
                        default='Co3d_v2')
    parser.add_argument("-ckpt", "--checkpoint", help="checkpoint number",
                        default=None)
    parser.add_argument("-ds", "--dataset", help="Choose dataset",
                        default='co3d_v2')

    return parser

parser = config_parser()
args = parser.parse_args()

def train(): # Train
    print("Train Ray Diffusion")
    '''
    Loading Model (Diffusion / Regression)
    '''
    device = torch.device("cuda" if args.multigpu else 'cpu')
    model, cfg = load_model(args.op,
                            checkpoint=args.ckpt,
                            device=device)
    
    wandb.config({"learning rate" : cfg.training.lr,
                  "epoch" : cfg.training.max_iterations,
                  "batch_size" : cfg.training.batch_size,
                  "Model type" : "Diffusion"})
    
    '''
    Preparing dataset
    '''
    # Dataset configuration
    # 7Scenes / Cambridge / Co3d_v2
    ds = Co3dDataset(co3d_dir=args.dp,
                    co3d_annotation_dir=args.dp)
    
    dl = DataLoader(ds,
                    batch_size=cfg.training.batch_size,
                    num_workers=cfg.training.num_workers,
                    shuffle=True)
    '''
    Train
    '''
    # set loss
    loss_fn = nn.MSELoss()

    # set optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.training.lr)

    # Start train
    print('Training configuration -----------------------------')
    for k, v in cfg.items():
        print(k + ':')
        for kk, vv in v.items():
            print('\t',kk,':',vv)
    print('----------------------------------------------------')

    for epoch in range(cfg.training.max_itertations):
        print("Epoch : {epoch}".format(epoch=epoch),"----------")
        for batch_idx, batch in enumerate(dl):
            # Generate GT Ray

            # Generate Predicted Ray

            # Calculate Loss
            loss = loss_fn(gt_ray, pred_ray) # According to Section 3.2 & 3.3

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            wandb.log({"Training loss" : loss.item()})

            
if __name__ == '__main__':
    train()