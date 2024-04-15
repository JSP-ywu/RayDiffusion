import os.path as osp
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from pytorch3d.renderer import PerspectiveCameras

from ray_diffusion.dataset.co3d_v2 import Co3dDataset
from ray_diffusion.dataset.custom import CustomDataset

from ray_diffusion.utils.rays import cameras_to_rays, Rays

#from ray_diffusion.model.diffuser import RayDiffuser
#from ray_diffusion.model.dit import Dit
#from ray_diffusion.model.feature_extractors import SpatialDino
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras

# Custom params
CUSTOM_KEY = None
HOME = '/home/vision/pjs/RayDiffusion'

import wandb
with open('./wandb_key.txt', 'r') as f: # Use your own authorized key
    wandb_key = f.readline()
wandb.login(key=wandb_key)
wandb.init(project="RayDiffusion - compute_x0")

'''
Parsing Argument
'''
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--datapath", help="dataset path",
                        default='/data1')
    parser.add_argument("-ap", '--annopath', help="annotation path",
                        default='/data1')
    parser.add_argument('-op', '--outputpath', help="output path",
                        default='/data1')
    parser.add_argument("-multigpu", action='store_true', help="whether use multiple gpu or not",
                        default=True)
    parser.add_argument("-ds", "--dataset", help="choose dataset <7scenes, Cambridge, Co3d_v2>",
                        default='Co3d_v2')
    parser.add_argument("-ckpt", "--checkpoint", help="checkpoint number",
                        default=None)
    parser.add_argument("-rn", "--run_name", help="Name of current run",
                        default='RayDiffusion_compute_x0')

    return parser

parser = get_parser()
args = parser.parse_args()

def train(): # Train
    print("Train Ray Diffusion")
    if not osp.exists(osp.join(args.outputpath, args.run_name)):
        os.makedirs(osp.join(HOME, args.outputpath, args.run_name))
    '''
    Loading Model (Diffusion / Regression)
    '''
    device = torch.device("cuda" if args.multigpu else 'cpu')
    model, cfg = load_model(args.outputpath,
                            checkpoint=args.checkpoint,
                            device=device)
    n_gpus = 8

    model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model.to(device)

    wandb.config.update({"learning rate" : cfg.training.lr,
                  "epoch" : cfg.training.max_iterations,
                  "batch_size" : cfg.training.batch_size,
                  "ngpus" : n_gpus,
                  "Model type" : "Diffusion"})
    
    '''
    Preparing dataset
    '''
    # Dataset configuration
    # 7Scenes / Cambridge / Co3d_v2
    ds = Co3dDataset(co3d_dir=args.datapath,
                    co3d_annotation_dir=args.annopath)
    


    dl = DataLoader(ds,
                    batch_size=cfg.training.batch_size,
                    num_workers=cfg.training.num_workers * n_gpus,
                    shuffle=True)
    '''
    Train
    '''
    # set loss / L2 loss
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
    epoch = 0
    cur_iter = 0
    while True:
        print("Epoch : {epoch}".format(epoch=epoch),"----------")
        for batch_idx, batches in enumerate(dl):
            images = batches['image'].to(device, non_blocking=True)
            # Generate GT Ray
            # (B x N) x H*W x 6
            print(images.size()) # B x N x C x H x W
            print(batches['R'].size()) # B x N x 3 x 3
            print(batches['T'].size()) # B x N x 3
            print(batches['crop_parameters'].size()) # B x N x 4
            org_rays = cameras_to_rays(cameras=PerspectiveCameras(R=batches['R'],
                                                                  T=batches['T'],
                                                                  device=device),
                                      crop_parameters=batches['crop_parameters'])
            print(org_rays.size())
            gt_rays = Rays.from_spatial(org_rays)[0]
            print(gt_rays.size())

            # Generate Predicted Ray
            # (B x N) x H*W x 6
            p_rays = predict_cameras(model=model,
                                     images=images,
                                     device=device,
                                     pred_x0=cfg.training.pred_x0,
                                     crop_parameters=batches['crop_parameters'],
                                     return_rays=True)[1]
            print(p_rays.size())
            pred_rays = Rays.from_spatial(p_rays)[0]
            print(pred_rays.size())
            
            # Calculate Loss
            loss = loss_fn(gt_rays, pred_rays) # According to Section 3.2 & 3.3
            # loss = loss_fn(gt_intrinsics, pred_intrinsics) # calculate loss using intrinsics

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            wandb.log({"Training loss" : loss.item()})
            if batch_idx % (int(len(dl) / 10)): # Print around 10 times per iteration
                print('Batch Iter[{batch_idx:04d}/{batch_len}]\t'.format(batch_idx=batch_idx, batch_len=len(dl)),
                      'Training loss : {loss:0.3f}'.format(loss.item()))

            # Save model
            if cur_iter % cfg.training.interval_save_checkpoint == 0:
                print('Save checkpoint {cur_iter:05d}'.format(cur_iter=cur_iter))
                ckpt_name = "ckpt_{cur_iter:05d}.pth.tar".format(cur_iter=cur_iter)
                save_dict = {"state_dict" : model.state_dict(),
                             "iteration" : cur_iter,
                             "epoch" : epoch,}
                torch.save(save_dict, osp.join(args.outputpath, args.run_name, ckpt_name))

            cur_iter = cur_iter + 1

        epoch = epoch + 1
        
        # Checking maximum iterations
        if cur_iter >= cfg.training.max_iterations:
            print('Reach maximum iterations.')
            print('Finsh training.')
            break
    wandb.finish()

            
if __name__ == '__main__':
    train()