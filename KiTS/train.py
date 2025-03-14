# %%
import os
import random
from time import time
import datetime

import SimpleITK as sitk
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from config import GenConfig

from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss

from metric import dice

from model import UNet3D

# %%
opt = GenConfig()


# %% Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# %% Seed 고정

seed = 19

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

random.seed(seed)

np.random.seed(seed)

# %% GPU Setting  --> Only for single GPU

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Model setting

# GeneratorNet = getattr(import_module('model'), opt.model_name)
generator = UNet3D(in_channels = 1 , num_classes = 4)

generator = generator.to(device)

## If restarting
if opt.restart:
    generator.load_state_dict(torch.load(''))


# %% Optimizer & Loss Function

optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


if 'DICE+CE' in opt.loss_function: 
    loss = DiceCELoss(include_background = False, lambda_dice = 1.0, lambda_ce= 1.0)
    
elif 'DICE+FOCAL' in opt.loss_function: 
    loss = DiceFocalLoss(include_background = False, lambda_dice = 1.0, lambda_focal= 1.0)
    
    

# %% DataLoader
from dataset import ImageDataset

dataset = ImageDataset(opt)

dataset_indices = list(range(len(dataset)))

kf = KFold(n_splits=5, shuffle=True, random_state=seed)


# %% Train

dt_now = datetime.datetime.now()

for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices)):
    # 각 fold에 대한 train과 validation Subset 생성
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 각 fold에 대한 DataLoader 생성
    train_loader = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_subset, batch_size=opt.val_batch_size, shuffle=False, num_workers=8)
    
    
    
    for epoch in range(opt.epoch, opt.n_epochs):
        
        Loss_train_per_epoch = 0
        Dice_train_per_epoch = 0
        
        Loss_valid_ALL = 0
        Dice_valid_ALL = 0   
        
        for i, train_data in enumerate(train_loader):     # i  ---> batch, train_data   ----> 말 그대로 data
            
            train_start = time()
            
            train_input = train_data['X'].to(device).float()    # torch.Size([2, 1, 192, 512, 512])
            train_input.requires_grad_(True)
            
            # print(np.sum(train_input.cpu().detach().numpy()))
            
            train_label = train_data['Y'].to(device).float()      # torch.Size([2, 4, 192, 512, 512])
            train_name = train_data['patient_name']
            
            optimizer.zero_grad()
            
            train_output = generator(train_input)
            
            
            criterion = loss(train_output, train_label)
            
            criterion.backward()
            optimizer.step()
            
            train_end = time()
            
            
            train_output_np = train_output.cpu().detach().numpy()
            train_label_np = train_label.cpu().detach().numpy()
            
            
            # print(np.sum(train_output_np))
            
            train_output_np[train_output_np>0.5] = 1
            train_output_np[train_output_np<=0.5] = 0
            train_label_np = train_label_np.astype(np.int64)
            
            train_output_bool = train_output_np.astype(bool)
            train_label_bool = train_label_np.astype(bool)
            
            
            DSC = dice(train_output_bool, train_label_bool)
            

            
            Loss_train_per_epoch += criterion
            Dice_train_per_epoch += DSC
            
            print("[Fold %d/%d   Epoch %3d/%3d   Batch %3d/%3d] [Train loss: %.4f | DICE score: %.4f] takes %5.3f (s)."
                % (fold+1, 5, epoch+1, opt.n_epochs, i+1, len(train_loader), criterion.item(), DSC, (train_end - train_start)))
            
            writer.add_scalar("Loss/train", Loss_train_per_epoch.item()/(i+1), epoch+1)
            writer.add_scalar("DICE/train", Dice_train_per_epoch/(i+1), epoch+1)
            
        
        ## Validation
        
        with torch.no_grad():
            
            generator.eval()
            
            best_dice = 0
            
            for v, valid_data in enumerate(val_loader):
                
                valid_start = time()
                
                valid_input = valid_data['X'].to(device).to(torch.float)
                valid_label = valid_data['Y'].to(device).to(torch.float)
                valid_name = valid_data['patient_name']
                
                valid_output = generator(valid_input)
                
                cri = loss(valid_output, valid_label)
                
                valid_end = time()
                
                
                valid_output_np = valid_output.cpu().detach().numpy()
                valid_label_np = valid_label.cpu().detach().numpy()
                
                valid_output_np[valid_output_np>0.5] = 1
                valid_output_np[valid_output_np<=0.5] = 0
                valid_label_np = valid_label_np.astype(np.int64)
                
                valid_output_bool = valid_output_np.astype(bool)
                valid_label_bool = valid_label_np.astype(bool)
                
                
                DSCore = dice(valid_output_bool, valid_label_bool)
                
                
                Loss_valid_ALL += cri
                Dice_valid_ALL += DSCore
                
                
                
                print("[Batch %3d/%3d] [Val loss: %.4f | DICE score: %.4f] takes %5.1f (s)."
                % (v+1, len(val_loader), cri.item(), DSCore, (valid_end - valid_start)))
                
                
                if DSCore > best_dice:
                    
                    torch.save(generator.state_dict(), (opt.model_save_dir+str(dt_now)+'_'+opt.loss_function+'_generator_%d.pth') % (epoch + 1))
                    best_dice = DSCore
                
                
                writer.add_scalar("Loss/valid", Loss_valid_ALL.item()/(v+1), epoch+1)
                writer.add_scalar("DICE/valid", Dice_valid_ALL/(v+1), epoch+1)
                
                
                
        
                
                
writer.close()