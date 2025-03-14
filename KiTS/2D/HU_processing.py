import SimpleITK as sitk
import numpy as np

import time
import os

from config import GenConfig



opt = GenConfig()

def HU_processing(ct_array, clip_min, clip_max):
    
    # start_time = time.time()
    
    # Load the CT data using SimpleITK
    # file_name = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
    # ct_image = sitk.ReadImage(file_name)
    
    # # Convert the SimpleITK image to a numpy array
    # ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

    # Image Processing ------------------------------
    HU_air = -1000
    th_11 = -1700
    th_12 = -1300
    th_21 = -700
    th_22 = -300

    if np.min(ct_array)>=-1024: # case (a)
        ct_array = ct_array + HU_air - np.min(ct_array)
    elif np.sum((ct_array>th_11) & (ct_array<th_12))==0 and np.sum((ct_array>th_21) & (ct_array<th_22))==0: # case (b1), (b2)
        ct_array = ct_array + HU_air
    elif np.sum((ct_array>th_11) & (ct_array<th_12))==0 and np.sum((ct_array>th_21) & (ct_array<th_22))!=0: # case (c1), (c2)
        pass
    elif np.sum((ct_array>th_11) & (ct_array<th_12))!=0:
        ct_array = ct_array - HU_air
    else:
        pass

    ct_array = np.clip(ct_array, clip_min, clip_max)
    
    
      
                 
    return ct_array
              
# from multiprocessing.pool import Pool
    
    
# if __name__ == "__main__":
    
#     data_dir = opt.data_dir
#     save_dir = ''
#     patient_names = os.listdir(data_dir)
#     patient_names.sort()   
    
#     clip_min = opt.clip_min
#     clip_max = opt.clip_max
    
#     args_list = [(data_dir, patient_name, save_dir, clip_min, clip_max) for patient_name in patient_names]   


#     pool = Pool(processes=20)
    
#     pool.starmap(preprocessing, args_list)


#     pool.close()
#     pool.join()
    