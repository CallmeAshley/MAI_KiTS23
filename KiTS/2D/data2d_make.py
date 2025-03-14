import numpy as np
import SimpleITK as sitk
import os

import shutil

from time import time

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import os

from HU_processing import HU_processing
from prepro import preprocessing

from multiprocessing.pool import Pool






def toslice(data_dir, patient_name, save_dir):
    
    
    X_file_dir = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
    Y_file_dir = os.path.join(data_dir, patient_name, 'segmentation.nii.gz')
    
    file_name = os.listdir(save_dir)
    
    start = time()
    seg_image = sitk.ReadImage(Y_file_dir)
    
    
    if any(patient_name in element for element in file_name) == False:

        ct_image = sitk.ReadImage(X_file_dir)
        original_spacing = ct_image.GetSpacing()
        original_direction = ct_image.GetDirection()
        original_origin = ct_image.GetOrigin()


        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float64)


        
        # seg_array = sitk.GetArrayFromImage(seg_image).astype(np.int64)

        # load_time = time()

        ct_array = HU_processing(ct_array, clip_min=-150, clip_max=800)
        ct_image = sitk.GetImageFromArray(ct_array)    # -----> array에는 spacing에 대한 정보가 없으므로 임의로 1,1,1로 맞춰짐.
        ct_image.SetSpacing(original_spacing)
        ct_image.SetDirection(original_direction)
        ct_image.SetOrigin(original_origin)


        # if self.is_valid==False and self.is_test==False: # train일 경우  preprocessing
        ct_array = preprocessing(ct_image, is_img = True, is_seg = False)
        seg_array = preprocessing(seg_image, is_img = False, is_seg = True)        # HWD로 표현
        # seg_array = seg_array.astype(np.int64)
        
        
        # if not os.path.exists(save_dir+patient_name):
        #     os.makedirs(save_dir+patient_name)
        
        
        for i in range(ct_array.shape[-1]):
            np.save(save_dir+'/img_'+patient_name+'_'+str(i)+'.npy', np.float64(ct_array[:,:,i]))
            np.save(save_dir+'/seg_'+patient_name+'_'+str(i)+'.npy', np.uint8(seg_array[:,:,i]))
            
        end1 = time()
         
        print(patient_name+'is Done! takes %s(s).'%(end1-start))
        
    else:
        
        count = sum(patient_name in element for element in file_name)
        
        if count != seg_image.GetSize()[0]:
            
            ct_image = sitk.ReadImage(X_file_dir)
            original_spacing = ct_image.GetSpacing()
            original_direction = ct_image.GetDirection()
            original_origin = ct_image.GetOrigin()


            ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float64)


            
            # seg_array = sitk.GetArrayFromImage(seg_image).astype(np.int64)

            # load_time = time()

            ct_array = HU_processing(ct_array, clip_min=-150, clip_max=800)
            ct_image = sitk.GetImageFromArray(ct_array)    # -----> array에는 spacing에 대한 정보가 없으므로 임의로 1,1,1로 맞춰짐.
            ct_image.SetSpacing(original_spacing)
            ct_image.SetDirection(original_direction)
            ct_image.SetOrigin(original_origin)


            # if self.is_valid==False and self.is_test==False: # train일 경우  preprocessing
            ct_array = preprocessing(ct_image, is_img = True, is_seg = False)
            seg_array = preprocessing(seg_image, is_img = False, is_seg = True)        # HWD로 표현
            # seg_array = seg_array.astype(np.int64)
            
            
            # if not os.path.exists(save_dir+patient_name):
            #     os.makedirs(save_dir+patient_name)
            
            
            for i in range(ct_array.shape[-1]):
                np.save(save_dir+'/img_'+patient_name+'_'+str(i)+'.npy', np.float64(ct_array[:,:,i]))
                np.save(save_dir+'/seg_'+patient_name+'_'+str(i)+'.npy', np.uint8(seg_array[:,:,i]))
        

            end2 = time()
            print(patient_name+'is Done! takes %s(s).'%(end2-start))
    
    
def fileswt(data_dir, patient_name, save_dir):
    
    file_dir = os.path.join(data_dir)
    
    all_files = os.listdir(file_dir)


    img_files = [f for f in all_files if f.startswith('img')]
    seg_files = [f for f in all_files if f.startswith('seg')]
    
    # if any(patient_name in element for element in img_files):
    
        # for img_file_name in img_files:
        #     img_file = os.path.join(file_dir, elemnet)
        #     shutil.move(img_file, save_dir)
        
        # for seg_file_name in seg_files:
        #     seg_file = os.path.join(file_dir, seg_file_name)
        #     shutil.move(seg_file, save_dir)
    for j in seg_files:
        
        seg_file = os.path.join(file_dir, j)
        seg = np.load(seg_file)
        if np.all(seg==0):
            shutil.move(seg_file, save_dir)
                
            img_file = os.path.join(file_dir, 'img'+j[3:])
            
            shutil.move(img_file, save_dir)
            
        
        
   
    
        

if __name__ == "__main__":
    
    data_dir = '/mai_nas/LSH/Data/KiTS_2D/'
    patient_names = os.listdir(data_dir)
    patient_names.sort() 
    
    # patient_names = ['case_00001', 'case_00003', 'case_00014', 'case_00015', 'case_00041', 'case_00042', 'case_00050', 'case_00060', 'case_00080', 'case_00098', 'case_00101', 'case_00107', 'case_00128', 'case_00136', 'case_00156', 'case_00166', 'case_00178', 'case_00191', 'case_00192', 'case_00199', 'case_00209', 'case_00212', 'case_00214', 'case_00231', 'case_00235', 'case_00241', 'case_00244', 'case_00249', 'case_00259', 'case_00270', 'case_00278', 'case_00284', 'case_00286', 'case_00295', 'case_00297', 'case_00298', 'case_00402', 'case_00413', 'case_00418', 'case_00437', 'case_00445', 'case_00464', 'case_00481', 'case_00512', 'case_00548', 'case_00558', 'case_00567', 'case_00571', 'case_00579']
    
    save_dir = '/mai_nas/LSH/Data/KiTS_bin'
    
    
    args_list = [(data_dir, patient_name, save_dir) for patient_name in patient_names]
    
    
    
    pool = Pool(processes=16)
    
    
    pool.starmap(fileswt, args_list)
    
    pool.close()
    pool.join()