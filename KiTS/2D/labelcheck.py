import numpy as np
import SimpleITK as sitk
import cv2

from time import time

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import os



data_dir = '/mai_nas/LSH/Data/KiTS_2D/'

data_list = os.listdir(data_dir)

seg_files = [f for f in data_list if f.startswith('seg')]
seg_files.sort()

for i in seg_files:
    
    seg_array = np.load(data_dir+i)
    # seg_array = cv2.resize(seg_array, (256,256))
    
    if np.all(seg_array == 0):
        print(i)

    # transform = torchvision.transforms.ToTensor()

    # Y = transform(seg_array).to(torch.long)
    # Y = torch.permute(Y, (0,2,1))
    # Y = F.one_hot(Y, num_classes = 4).squeeze()    
    # Y = torch.permute(Y, (2,1,0))
    
    
    
        
        
