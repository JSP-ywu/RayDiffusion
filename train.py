import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import configargparse
from pathlib import Path

from ray_diffusion.dataset.co3d_v2 import Co3dDataset
from ray_diffusion.dataset.custom import CustomDataset

#from ray_diffusion.model.diffuser import RayDiffuser
#from ray_diffusion.model.dit import Dit
#from ray_diffusion.model.feature_extractors import SpatialDino
from ray_diffusion.inference.load_model import load_model

from ray_diffusion.model.diffuser import RayDiffuser
from ray_diffusion.model.scheduler import NoiseScheduler

# Custom Key for training configuration
CUSTOM_KEY = None

'''
Parsing Argument
'''
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-dp", "--datapath", help="dataset path", default='/data1')
    parser.add_argument('-op', '--outputpath', help="output path", default='/data1')
    parser.add_argument("-multigpu", action='store_ture', help="whether use multiple gpu or not", default=True)
    parser.add_argument("-ds", "--dataset", help="choose dataset", default='co3d')
    parser.add_argument("-ckpt", "--checkpoint", help="checkpoint number", default=None)

    return parser

parser = config_parser()
args = parser.parse_args()

def train(): # Train
    print("Train Ray Diffusion")
    '''
    Loading Model (Diffusion / Regression)
    '''
    device = torch.device("cuda" if args.multigpu else 'cpu')
    model, cfg = load_model(args.op, checkpoint=args.ckpt, device=device)

    '''
    Preparing dataset
    '''
    # Dataset configuration
    dt_loader = CustomDataset(args.dp, None, None, None) # return image batches
    
    '''
    Train
    '''
    # set loss
    loss = nn.MSELoss()

    # set optimizer
    optimizer = optim.Adam(model.parameters(), )

if __name__ == '__main__':
    train()