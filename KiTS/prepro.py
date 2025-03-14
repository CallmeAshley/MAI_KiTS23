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
        normalized_data = stats.zscore(resampled_array, axis=None)  #z-score normalization
        # normalized_data = (resampled_array - resampled_array.min()) / (resampled_array.max() - resampled_array.min())    # min max norm
        
        return normalized_data
    
    if is_seg:

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resampler.Execute(data)
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        
        return resampled_array
 
        
        
    # non-zero crop   ---> 후에 추가 예정
    


def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    
    return np.pad(image[slicer], to_padding, **kwargs)



