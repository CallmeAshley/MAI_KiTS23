import numpy as np
import SimpleITK as sitk
import os
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import time

from HU_processing import HU_processing

import scipy.stats as stats
from scipy.ndimage import zoom


# def plot_hist(data_dir, patient_name, save_dir, color='b',  range=[-100, 500]):
    
    
    
#     start_time = time.time()
    
#     # Load the CT data using SimpleITK
#     file_name = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
#     ct_image = sitk.ReadImage(file_name)
    
#     # label_dir = os.path.join(data_dir, patient_name, 'segmentation.nii.gz')
#     # label = sitk.ReadImage(label_dir)

#     # Convert the SimpleITK image to a numpy array
#     ct_array = sitk.GetArrayFromImage(ct_image)
#     # label_array = sitk.GetArrayFromImage(label)
    
#     # vessel_pixels = ct_array[label_array!=0]

#     # Flatten the array to get a 1D array of HU values
#     # hu_values = vessel_pixels.flatten()
    
#     ct_array = HU_processing(ct_array, -150, 800)
    
#     # d, h, w = 96, 128, 128   # 정해야 함
    
#     # 새로운 크기에 대한 비율 계산 
#     # z_ratio = d / ct_array.shape[2]
#     # y_ratio = w / ct_array.shape[1]
#     # x_ratio = h / ct_array.shape[0]
    
#     # resampled_data_array = zoom(ct_array, (x_ratio, y_ratio, z_ratio), order=1)   # 선형 보간   
#     normalized_data = stats.zscore(ct_array, axis=None)  #z-score normalization
    
#     hu_values = normalized_data.flatten()
    
#     range=[int(hu_values.min())-10, int(hu_values.max())+10]
    

#     # Plot the histogram
#     counts, bins = np.histogram(hu_values, bins=100, range=range)
#     b, bins, patches = plt.hist(bins[:-1], bins=100, range=range, weights=np.log(np.array(counts)+1), color=color)
#     plt.xlim(range)
#     plt.xlabel('Hounsfield Units (HU)')
#     plt.ylabel('Frequency')
#     plt.text(int(hu_values.min()), int(np.log(np.array(counts)+1).max()),
#              'Avg: ' + str(round(np.average(hu_values), 2)) + '   Std: ' + str(round(np.std(hu_values), 2)) + '    min: ' + str(round(np.min(hu_values), 0))  + '    max: ' +  str(round(np.max(hu_values), 0)),
#              ha='left', va='top')
    
#     plt.title(patient_name+'_zscore')
    
#     # if not os.path.exists(os.path.join(save_dir, patient_name)):
#     #     os.makedirs(os.path.join(save_dir, patient_name))

#     plt.savefig(save_dir + str(patient_name)+'_zscore'+ '.png')
#     plt.close()
    
#     print('Done! %s takes %s seconds.'%(patient_name, (time.time()-start_time)))
    
    
    
    
def spacing(data_dir, patient_name):
    
    file_name = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
    ct_image = sitk.ReadImage(file_name)
    
    # hu_values = ct_image.flatten()
    value = ct_image.GetSpacing() 
    # min_value=[]
    # max_value=[]
    
    # min_value += str(round(np.min(hu_values), 0))
    # max_value += str(round(np.max(hu_values), 0))
    
    # f = open(save_dir+patient_name+'/seg_hu.txt', 'w')
    # f.write(str(value))
    
    # f.close()
    
    # print('%s is Done!'%patient_name)
    
    return value[0], value[1], value[2]     # z, x, y로 반환



def size(data_dir, patient_name):
    
    file_name = os.path.join(data_dir, patient_name, 'imaging.nii.gz')
    ct_image = sitk.ReadImage(file_name)
    
    img_size = ct_image.GetSize()
    
    return img_size[0], img_size[1], img_size[2]   # z, x, y
    
    
    

if __name__ == "__main__":

    data_dir = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23/dataset/'
    # save_dir = '/mai_nas/LSH/KiTS_analysis/after_HU_processing/'
    patient_names = os.listdir(data_dir)
    patient_names.sort()     
    
    z_space = []
    x_space = []
    y_space = []
    
    for i in patient_names:
        z,x,y = size(data_dir, i)
        z_space.append(z)
        x_space.append(x)
        y_space.append(y)
        
    x_space=np.array(x_space)
    y_space=np.array(y_space)
    z_space=np.array(z_space)
        
    print('x avg: %.2f, y avg: %.2f, z avg: %.2f'%(np.average(x_space),np.average(y_space),np.average(z_space)))
    
    # args_list = [(data_dir, patient_name, save_dir) for patient_name in patient_names]   


    # pool = Pool(processes=16)
    
    # # pool.starmap(plot_hist, args_list)
    # pool.starmap(spacing, args_list)
    
    # pool.close()
    # pool.join()
    
    
        
# x = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23/dataset/case_00000/'

# label = sitk.ReadImage(x+'segmentation.nii.gz')
# img = sitk.ReadImage(x+'imaging.nii.gz')

# ct_array = sitk.GetArrayFromImage(img)
# label_array = sitk.GetArrayFromImage(label)

# vessel_pixels = ct_array[label_array!=0]


# print('dd')

          

#%% Doohyun 선배 

# img =  sitk.ReadImage(img_path)
# img =  sitk.GetArrayFromImage(img).astype(np.float32)
# mask =  sitk.ReadImage(mask_path)
# mask =  sitk.GetArrayFromImage(mask)

# # Image Processing ------------------------------
# HU_air = -1000
# th_11 = -1700
# th_12 = -1300
# th_21 = -700
# th_22 = -300

# if np.min(img)>=-1024: # case (a)
#     img = img + HU_air - np.min(img)
# elif np.sum((img>th_11) & (img<th_12))==0 and np.sum((img>th_21) & (img<th_22))==0: # case (b1), (b2)
#     img = img + HU_air
# elif np.sum((img>th_11) & (img<th_12))==0 and np.sum((img>th_21) & (img<th_22))!=0: # case (c1), (c2)
#     pass
# elif np.sum((img>th_11) & (img<th_12))!=0:
#     img = img - HU_air
# else:
#     pass

# clip_min = -150
# clip_max = 800


# img = np.clip(img,clip_min,clip_max)
            
            
            
# %%  text
# data_dir = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23/dataset/'
# save_dir = '/mai_nas/LSH/KiTS_analysis/HU/'
# patient_name = os.listdir(data_dir)
# patient_name.sort()  

# f = open(save_dir+'/seg_hu.txt', 'w')

# for i in patient_name:
  
#     file_name = os.path.join(data_dir, i, 'imaging.nii.gz')
#     ct_image = sitk.ReadImage(file_name)
#     ct_array = sitk.GetArrayFromImage(ct_image)
    
#     label_dir = os.path.join(data_dir, i, 'segmentation.nii.gz')
#     label = sitk.ReadImage(label_dir)
#     label_array = sitk.GetArrayFromImage(label)
    
#     vessel_pixels = ct_array[label_array!=0]

#     # Flatten the array to get a 1D array of HU values
#     hu_values = vessel_pixels.flatten()
#     # hu_values = ct_array.flatten()



#     min_value = str(round(np.min(hu_values), 0))
#     max_value = str(round(np.max(hu_values), 0))
    
    
    
#     f.write(i+'    '+str(min_value)+'    '+str(max_value)+'\n')
#     f.close()
    
#     f = open(save_dir+'/seg_hu.txt', 'a')
    
#     print('%s is Done!'%i)


# f.close()


