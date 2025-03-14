import SimpleITK as sitk
import numpy as np

import time
import os



import scipy.stats as stats

# import torch
# import torch.nn.functional as F

from config import GenConfig

opt = GenConfig()

def preprocessing(data, is_img = False, is_seg = False):        # 64 512 512      (5.0, 0.921875, 0.921875)
    
    # data = sitk.GetImageFromArray(data)
    
    original_size = data.GetSize()
    original_spacing = data.GetSpacing()
    
    # voxel size resampling
    out_spacing = np.array([3.36, 0.8, 0.8])     # 정해야 함
    
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    
    # d, h, w = 116, 132, 132  # 정해야 함
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    # resampler.SetSize((d, h, w))  
    # resampler.SetSize((data.GetWidth() // 2, data.GetHeight() // 2, data.GetDepth() // 2))
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(data.GetDirection())
    resampler.SetOutputOrigin(data.GetOrigin())
    # resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(data.GetPixelIDValue())
    
    


    # 리사이즈 수행
    if is_img:
  
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampled_image = resampler.Execute(data)
        
        # resampled_spacing = resampled_image.GetSpacing()
        # resampled_size = resampled_image.GetSize()
        # resampled_direction = resampled_image.GetDirection()
        # resampled_origin = resampled_image.GetOrigin()
        
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        normalized_data = stats.zscore(resampled_array, axis=None)  #z-score normalization      H,W,D
        # normalized_data = (resampled_array - resampled_array.min()) / (resampled_array.max() - resampled_array.min())    # min max norm
        
        return normalized_data      
    
    if is_seg:

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resampler.Execute(data)
        resampled_array = sitk.GetArrayFromImage(resampled_image)     #  H,W,D
        
        return resampled_array