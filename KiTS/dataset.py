import numpy as np
import SimpleITK as sitk

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

opt = GenConfig()


class ImageDataset(Dataset):
    def __init__(self, opt, is_valid=False, is_test=False):
        self.opt = opt
        
        self.clip_min = opt.clip_min
        self.clip_max = opt.clip_max
        
        self.data_dir = opt.data_dir
        
        self.patient_name = os.listdir(self.data_dir)
        self.patient_name.sort()      # list로 patient_name 반환
        
        # if 'KiTS' in self.opt.dataset:
            # if is_valid:

            # elif is_test:
            #     self.X_file_dir = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
            #     self.Y_file_dir = os.path.join(data_dir, patient_name, 'segmentation.nii.gz')
            # else:
            #     self.X_file_dir = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
            #     self.Y_file_dir = os.path.join(data_dir, patient_name, 'segmentation.nii.gz')
        
        self.is_valid = is_valid
        self.is_test = is_test
        self.transform = torchvision.transforms.ToTensor()
        
    def __getitem__(self, index):
        
        start = time()

        self.X_file_dir = os.path.join(self.data_dir, self.patient_name[index], 'imaging.nii.gz')
        self.Y_file_dir = os.path.join(self.data_dir, self.patient_name[index], 'segmentation.nii.gz')
        
        ct_image = sitk.ReadImage(self.X_file_dir)
        original_spacing = ct_image.GetSpacing()
        original_direction = ct_image.GetDirection()
        original_origin = ct_image.GetOrigin()

        
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float64)
        
        
        seg_image = sitk.ReadImage(self.Y_file_dir)
        # seg_array = sitk.GetArrayFromImage(seg_image).astype(np.int64)
        
        # load_time = time()
        
        ct_array = HU_processing(ct_array, self.clip_min, self.clip_max)
        ct_image = sitk.GetImageFromArray(ct_array)    # -----> array에는 spacing에 대한 정보가 없으므로 임의로 1,1,1로 맞춰짐.
        ct_image.SetSpacing(original_spacing)
        ct_image.SetDirection(original_direction)
        ct_image.SetOrigin(original_origin)

        
        # if self.is_valid==False and self.is_test==False: # train일 경우  preprocessing
        ct_array = preprocessing(ct_image, is_img = True, is_seg = False)
        seg_array = preprocessing(seg_image, is_img = False, is_seg = True)
        seg_array = seg_array.astype(np.int64)
        
        
       # ---------------------- voxel spacing 맞춰줘야 함 ------------------------------- 
        
        
        
        # if self.is_valid: 
                      
        X = self.transform(ct_array)   # output --> (D, H, W)
        X = X.unsqueeze(0)    
        
        Y = self.transform(seg_array)
        Y = F.one_hot(Y, num_classes = 4)
        Y = torch.permute(Y, (3, 0, 1, 2))
        
        
        # print('Time taken to load the data: %5.1f (s)'%(load_time - start))
        

        return {'X': X, 'Y': Y, 'patient_name': self.patient_name[index]}

    def __len__(self):
        
        return len(self.patient_name)