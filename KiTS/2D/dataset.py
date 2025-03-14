import numpy as np
import SimpleITK as sitk
import cv2
import random

from time import time

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import os

from HU_processing import HU_processing
from prepro import preprocessing
from config import GenConfig

from utils import elastic_transform2D

opt = GenConfig()


class ImageDataset(Dataset):
    def __init__(self, opt, is_valid=False, is_test=False):
        self.opt = opt
        self.is_valid = is_valid
        self.is_test = is_test
        
        self.data_dir = opt.data_dir
    
        self.file_name = os.listdir(self.data_dir)
        self.file_name.sort()
        
        self.img_files = [f for f in self.file_name if f.startswith('img')]
        self.img_files.sort()
        
        self.seg_files = [f for f in self.file_name if f.startswith('seg')]
        self.seg_files.sort()
        
        self.bin_dir = '/mai_nas/LSH/Data/KiTS_bin/'
        self.file_bin = os.listdir(self.bin_dir)
        self.file_bin.sort()
        self.bin_img = [f for f in self.file_bin if f.startswith('img')]
        self.bin_img.sort()
        
        self.bin_seg = [f for f in self.file_bin if f.startswith('seg')]
        self.bin_seg.sort()
        
        
        # self.transform = torchvision.transforms.ToTensor()
        
    def __getitem__(self, index):
        
        start = time()
        
        self.img_files = [f for f in self.file_name if f.startswith('img')]
        self.img_files.sort()
        
        self.seg_files = [f for f in self.file_name if f.startswith('seg')]
        self.seg_files.sort()
        
        self.bin_img = [f for f in self.file_bin if f.startswith('img')]
        self.bin_img.sort()
        
        self.bin_seg = [f for f in self.file_bin if f.startswith('seg')]
        self.bin_seg.sort()
        
        
        
        # if 1 in seg_array or 2 in seg_array or 3 in seg_array:
        if random.choice([0, 1]):
            ct_array = np.load(self.data_dir+self.img_files[index]).astype(np.float32)
            ct_array = cv2.resize(ct_array, (256,256))
            
            seg_array = np.load(self.data_dir+self.seg_files[index]).astype(np.uint8)
            seg_array = cv2.resize(seg_array, (256,256))
            seg_array = np.transpose(seg_array, (1,0))
            
        else:
            ct_array = np.load(self.bin_dir+self.bin_img[index]).astype(np.float32)
            ct_array = cv2.resize(ct_array, (256,256))
            
            seg_array = np.load(self.bin_dir+self.bin_seg[index]).astype(np.uint8)
            seg_array = cv2.resize(seg_array, (256,256))
            seg_array = np.transpose(seg_array, (1,0))

        # seg_array = np.expand_dims(seg_array, 2)
        
        
    
        if self.is_valid==False and self.is_test==False: # train일 경우
            if random.choice([0, 1]):
                angleRot = int(np.random.rand(1)*30)
                warpMatrix = cv2.getRotationMatrix2D((seg_array.shape[0]//2, seg_array.shape[1]//2), angleRot, 1)  # (중심좌표, 반시계방향으로회전각, 확대비율)
                ct_array = cv2.warpAffine(ct_array, warpMatrix, (256, 256))
                seg_array = cv2.warpAffine(seg_array, warpMatrix, (256, 256))                
                
                
            # if random.choice([0,1]):
            #     ct_array, seg_array = elastic_transform2D(ct_array, seg_array)
        
        ct_array = np.expand_dims(ct_array, 2)
        # seg_array = np.squeeze(seg_array)
        
        X = torch.tensor(ct_array)   
        X = torch.permute(X, (2,1,0))
        # X = X.unsqueeze(0)    
        
        # Y = self.transform(seg_array).to(torch.long)
        # Y = F.one_hot(Y, num_classes = 4)     # 원본 텐서가 2차원인 경우, 원-핫 인코딩 후에는 3차원 텐서가 됩니다.
        # Y = Y.squeeze()
        # Y = torch.permute(Y, (2,1,0))
        Y = torch.tensor(seg_array).to(torch.long)
        # Y = torch.permute(Y, (1,0))
        Y = F.one_hot(Y, num_classes = 4)  
        Y = torch.permute(Y, (2,0,1))
        

        

        return {'X': X, 'Y': Y, 'patient_name': self.img_files[index]}
        
        # else:
        #     return self.__getitem__(index + 1)
        

    def __len__(self):
        
        return len(self.img_files)