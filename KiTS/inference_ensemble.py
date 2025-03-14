import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import torch
import torch.nn.functional as F

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

from typing import Tuple, Union, List



KITS_HEC_LABEL_MAPPING = {
    'kidney_and_mass': (1, 2, 3),
    'mass': (2, 3),
    'tumor': (2, ),
}

KITS_LABEL_TO_HEC_MAPPING = {j: i for i, j in KITS_HEC_LABEL_MAPPING.items()}

HEC_NAME_LIST = list(KITS_HEC_LABEL_MAPPING.keys())

# just for you as a reference. This tells you which metric is at what index.
# This is not used anywhere
METRIC_NAME_LIST = ["Dice", "SD"]

LABEL_AGGREGATION_ORDER = (1, 3, 2)
# this means that we first place the kidney, then the cyst and finally the
# tumor. The order matters! If parts of a later label (example tumor) overlap
# with a prior label (kidney or cyst) the prior label is overwritten

KITS_LABEL_NAMES = {
    1: "kidney",
    2: "tumor",
    3: "cyst"
}

# values are determined by kits21/evaluation/compute_tolerances.py
HEC_SD_TOLERANCES_MM = {
    'kidney_and_mass': 1.0330772532390826,
    'mass': 1.1328796488598762,
    'tumor': 1.1498198361434828,
}

# this determines which reference file we use for evaluation
GT_SEGM_FNAME = 'segmentation.nii.gz'

# how many groups of sampled segmentations?
NUMBER_OF_GROUPS = 5










def dice(prediction: np.ndarray, reference: np.ndarray):
    """
    Both predicion and reference have to be bool (!) numpy arrays. True is interpreted as foreground, False is background
    """
    intersection = np.count_nonzero(prediction & reference)
    numel_pred = np.count_nonzero(prediction)
    numel_ref = np.count_nonzero(reference)
    if numel_ref == 0 and numel_pred == 0:
        return np.nan
    else:
        return 2 * intersection / (numel_ref + numel_pred)



def construct_HEC_from_segmentation(segmentation: np.ndarray, label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Takes a segmentation as input (integer map with values indicating what class a voxel belongs to) and returns a
    boolean array based on where the selected label/HEC is. If label is a tuple, all pixels belonging to any of the
    listed classes will be set to True in the results. The rest remains False.
    """
    if not isinstance(label, (tuple, list)):
        return segmentation == label
    else:
        if len(label) == 1:
            return segmentation == label[0]
        else:
            mask = np.zeros(segmentation.shape, dtype=bool)
            for l in label:
                mask[segmentation == l] = True
            return mask
        
        
        

def compute_metrics_for_label(segmentation_predicted: np.ndarray, segmentation_reference: np.ndarray,
                              label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float],
                              sd_tolerance_mm: float = None) \
        -> Tuple[float, float]:
    """
    :param segmentation_predicted: segmentation map (np.ndarray) with int values representing the predicted segmentation
    :param segmentation_reference:  segmentation map (np.ndarray) with int values representing the gt segmentation
    :param label: can be int or tuple of ints. If tuple of ints, a HEC is constructed from the labels in the tuple.
    :param spacing: important to know for volume and surface distance computation
    :param sd_tolerance_mm
    :return:
    """
    assert all([i == j] for i, j in zip(segmentation_predicted.shape, segmentation_reference.shape)), \
        "predicted and gt segmentation must have the same shape"

    # make label always a tuple. Needed for inferring sd_tolerance_mm if not given
    label = (label,) if not isinstance(label, (tuple, list)) else label

    # build a bool mask from the segmentation_predicted, segmentation_reference and provided label(s)
    mask_pred = construct_HEC_from_segmentation(segmentation_predicted, label)
    mask_gt = construct_HEC_from_segmentation(segmentation_reference, label)
    gt_empty = np.count_nonzero(mask_gt) == 0
    pred_empty = np.count_nonzero(mask_pred) == 0

    if sd_tolerance_mm is None:
        sd_tolerance_mm = HEC_SD_TOLERANCES_MM[KITS_LABEL_TO_HEC_MAPPING[label]]

    if gt_empty and pred_empty:
        sd = 1
        dc = 1
    elif gt_empty or pred_empty:    # 여기로 들어와지는 것도 아님.
        sd = 0
        dc = 0
    else:
        dc = dice(mask_pred, mask_gt)
        dist = compute_surface_distances(mask_gt, mask_pred, spacing)
        sd = compute_surface_dice_at_tolerance(dist, tolerance_mm=sd_tolerance_mm)

    return dc, sd










def save_numpy_as_nifti(numpy_array, spacing, output_file_path):
    """
    Numpy 배열을 spacing 정보를 이용하여 NIfTI 파일로 저장하는 함수

    Parameters:
        numpy_array (np.ndarray): 저장할 Numpy 배열
        spacing (tuple): 각 축의 spacing 정보 (예: (1.0, 1.0, 1.0))
        output_file_path (str): 저장할 NIfTI 파일 경로

    Returns:
        None
    """
    # NIfTI 이미지 객체 생성
    nifti_img = nib.Nifti1Image(numpy_array, affine=np.eye(4))

    # spacing 정보를 이미지에 저장
    nifti_img.header.set_zooms(spacing)

    # NIfTI 파일로 저장
    nib.save(nifti_img, output_file_path)


# Pseudo Mask
pseudo_label_dir = '/mai_nas/YE/kits23/nnUNet/nnUNet_raw/Dataset220_KiTS2023/labelsTs/'
pseudo_label_list = os.listdir(pseudo_label_dir)
pseudo_label_list.sort()


#test_data 경로
ori_test_dir = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23_testdata/test_data/'

whole_crop_dir = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23_testdata/crop_img/'

#각 모델 npy 경로
nnunet_whole_crop_dir_3_best = '/mai_nas/YE/kits23/nnUNet/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/validation_npy/' ##whole crop infer dir
nunet_tumor_dir = '/mai_nas/YE/kits23/nnUNet/nnUNet_raw/inference_results_val0_tumor_only/tumor_only_npy/'#Tumor infer dir

nnformer_whole_dir = '/home/compu/WHY/whole_nnformer/'
nnformer_whole_crop_dir = '/home/compu/WHY/whole_crop_nnformer/'  #whole crop infer dir  

medformer_whole_crop_dir = '/mai_nas/LSH/infer_np_save/' ##whole crop infer dir

save_dir ='/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23_testdata/ensemble_result/'

name_list = sorted(os.listdir(nnunet_whole_crop_dir_3_best))
print("name_list : ", name_list)

dice_score123 = []
surface_score123 = []

dice_score23 = []
surface_score23 = []

dice_score2 = []
surface_score2 = []


for data_name in name_list:
    # print("data_name[:-4] : ", data_name[:-4])
    gt_mask_sitk = sitk.ReadImage(pseudo_label_dir + data_name[:-4]+'.nii.gz')
    gt_mask = sitk.GetArrayFromImage(gt_mask_sitk)     #xyz    (258, 119, 65)
     
    
    space = gt_mask_sitk.GetSpacing()    #zyx

    # space를 numpy 배열로 변환
    space_array = np.array(space)

    space_array = space_array[[2, 1, 0]]

    # 다시 tuple로 변환
    space_tuple = tuple(space_array)
    
    
    
    # BBox
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(pseudo_mask_sitk)
    
    bbox = labelShapeFilter.GetBoundingBox(1)

    # Bounding box information: [start_x, start_y, start_z, size_x, size_y, size_z]
    roi_filter = sitk.RegionOfInterestImageFilter()
    
    # Calculate new bbox start and size considering the limits of the image
    bbox_start_z = max(0, bbox[0]-10)  # make sure it is not less than 0
    bbox_size_z = min(pseudo_mask_sitk.GetSize()[0] - bbox_start_z, bbox[3]+20)  # make sure it is not beyond the image size

    bbox_start_y = max(0, bbox[1]-10)  
    bbox_size_y = min(pseudo_mask_sitk.GetSize()[1] - bbox_start_y, bbox[4]+20) 

    bbox_start_x = max(0, bbox[2]-10)  
    bbox_size_x = min(pseudo_mask_sitk.GetSize()[2] - bbox_start_x, bbox[5]+20)

    np_bbox = [[bbox_start_x, bbox_start_x+bbox_size_x],
            [bbox_start_y, bbox_start_y+bbox_size_y],
            [bbox_start_z, bbox_start_z+bbox_size_z]] # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    
    

    
    ori_test_data = sitk.ReadImage(ori_test_dir + data_name[:-4] + '.nii.gz')     ##원본 데이터
    ori_test_data_numpy = sitk.GetArrayFromImage(ori_test_data)
    
    # whole crop
    whole_crop_data = sitk.ReadImage(whole_crop_dir+data_name[:-4] + '.nii.gz')
    prepro_spacing = [3, 0.78125, 0.78125]
    nnformer_spacing = [1, 0.78125, 0.78125]

    print("data_name : ", data_name)
    ##nnUnet
    score_whole_crop_nnunet_3 = np.load(nnunet_whole_crop_dir_3_best + data_name)
    score_tumor_nnunet = np.load(nunet_tumor_dir + data_name)
    
    
    score_whole_crop_nnunet_3 = np.argmax(score_whole_crop_nnunet_3, axis=0)    # 추후 shape 보고
    score_whole_crop_nnunet_3 = np.transpose(score_whole_crop_nnunet_3, (2,1,0))
    score_whole_crop_nnunet_3 = np.uint8(score_whole_crop_nnunet_3)
    score_whole_crop_nnunet_3 = np.flip(score_whole_crop_nnunet_3, axis=(0, 1, 2))
    
    

    #nnFormer
    score_whole_nnformer = np.load(nnformer_whole_dir + data_name)
    score_whole_crop_nnformer = np.load(nnformer_whole_crop_dir + data_name[:-4]+'_0000.npy')
    
    ##medFormer
    score_whole_crop_medformer = np.load(medformer_whole_crop_dir + data_name)
    score_whole_medformer_transposed = np.transpose(score_whole_crop_medformer, (3, 0, 1, 2))
    #맞춰줘야하는 padding size 확인
    
    print("score_whole_crop_nnunet_3.shape : ", score_whole_crop_nnunet_3.shape)
    print("score_tumor_nnunet.shape : ", score_tumor_nnunet.shape)   
    # print("score_whole_crop_nnformer : ", score_whole_crop_nnformer.shape)
    print("set the original_size size : ", ori_test_data_numpy.shape)
    print("====================================================================")

    
    score_whole_crop_nnformer = torch.from_numpy(score_whole_crop_nnformer)[None,...]  # adds batch and channel dimensions
    # Interpolation을 수행합니다. 
    score_whole_crop_nnformer = F.interpolate(score_whole_crop_nnformer, size = score_whole_crop_nnunet_3.shape[1:], mode='trilinear', align_corners=False)
    score_whole_crop_nnformer= score_whole_crop_nnformer.squeeze().numpy()
    
    
    
    print("score_whole_crop_nnunet_3 : ", score_whole_crop_nnunet_3.shape)
    print("score_tumor_nnunet : ", score_tumor_nnunet.shape)
    
    a = np.ones_like(score_whole_crop_nnunet_3)
    a *= 0.7
    a[2] = 0     # tumor channel 다 0으로
    b = np.ones_like(score_tumor_nnunet)
    b = b*0.3
    b[2] = 1
    
    ensemble_array = a*score_whole_crop_nnunet_3 + b*score_tumor_nnunet 
    # ensemble_array = a*score_whole_crop_nnunet_3 + b*score_tumor_nnunet + c*score_whole_crop_nnformer 
    # + c*score_whole_crop_nnformer_padded + d*score_whole_nnformer + e*score_whole_medformer_padded
    print("ensemble_array : ", ensemble_array.shape)
    
    ensemble_result = np.argmax(ensemble_array, axis=0)    # 추후 shape 보고
    ensemble_result = np.transpose(ensemble_result, (2,1,0))
    ensemble_result = np.uint8(ensemble_result)
    ensemble_result = np.flip(ensemble_result, axis=(0, 1, 2))
    
    print("ensemble_result_class : ", np.unique(ensemble_result))
    print("ensemble_result_shape: ", ensemble_result.shape)
    
    score 계산 ####################################################################################################################
    
    hi_class0 = HEC_NAME_LIST[0]
    hi_class1 = HEC_NAME_LIST[1]
    hi_class2 = HEC_NAME_LIST[2]
    
    hi_class0 = KITS_HEC_LABEL_MAPPING[hi_class0]
    hi_class1 = KITS_HEC_LABEL_MAPPING[hi_class1]
    hi_class2 = KITS_HEC_LABEL_MAPPING[hi_class2]
    
    sd_tolerance_mm0 = HEC_SD_TOLERANCES_MM[KITS_LABEL_TO_HEC_MAPPING[hi_class0]]
    sd_tolerance_mm1 = HEC_SD_TOLERANCES_MM[KITS_LABEL_TO_HEC_MAPPING[hi_class1]]
    sd_tolerance_mm2 = HEC_SD_TOLERANCES_MM[KITS_LABEL_TO_HEC_MAPPING[hi_class2]]
    
    
    
    
    print('Patient_name: ', data_name[:-4])

    
    dc123,sd123 = compute_metrics_for_label(score_whole_crop_nnunet_3, gt_mask, label=hi_class0,spacing=space_tuple, sd_tolerance_mm=sd_tolerance_mm0)
    
    print("dice score: %.8f"%dc123)
    print("surface dice score: %.8f"%sd123)
    
    dice_score123.append(dc123)
    surface_score123.append(sd123)
    
    
    
    
    dc23,sd23 = compute_metrics_for_label(score_whole_crop_nnunet_3, gt_mask, label=hi_class1,spacing=space_tuple, sd_tolerance_mm=sd_tolerance_mm1)
    
    print("dice score: %.8f"%dc23)
    print("surface dice score: %.8f"%sd23)
    
    dice_score23.append(dc23)
    surface_score23.append(sd23)
    
    
    
    
    dc2,sd2 = compute_metrics_for_label(score_whole_crop_nnunet_3, gt_mask, label=hi_class2,spacing=space_tuple, sd_tolerance_mm=sd_tolerance_mm2)
    
    print("dice score: %.8f"%dc2)
    print("surface dice score: %.8f"%sd2)
    
    dice_score2.append(dc2)
    surface_score2.append(sd2)
    
    
    
dice_mean_123 = sum(dice_score123)/98
dice_mean_23 = sum(dice_score23)/98
dice_mean_2 = sum(dice_score2)/98

sur_mean_123 = sum(surface_score123)/98
sur_mean_23 = sum(surface_score23)/98
sur_mean_2 = sum(surface_score2)/98


print("dice score mean_123: %.8f"%dice_mean_123)
print("surface dice score mean_123: %.8f"%sur_mean_123)

print("dice score mean_23: %.8f"%dice_mean_23)
print("surface dice score mean_23: %.8f"%sur_mean_23)

print("dice score mean_2: %.8f"%dice_mean_2)
print("surface dice score mean_2: %.8f"%sur_mean_2)
    
    
    
    ######################################################################################################################
    
print("original_size[0]:", original_sisze[0])
print("ensemble_result[0]:", ensemble_result.shape[0])

X,Y,Z = ori_test_data_numpy.shape
target_shape = ensemble_result.shape
x,y,z = ensemble_result.shape


# Padding 적용
padded_result = np.zeros_like(ori_test_data_numpy)
ensemble_result = ensemble_result.transpose(2, 1, 0)
ensemble_result = np.flip(ensemble_result, axis=(0, 1, 2))
padded_result[np_bbox[0][0]:np_bbox[0][1], np_bbox[1][0]:np_bbox[1][1], np_bbox[2][0]:np_bbox[2][1]] = ensemble_result
padded_result = np.pad(ensemble_result, 
                    pad_width=((pad_amounts[0], pad_amounts[0] + pad_amounts_odd[0]),
                                (pad_amounts[1], pad_amounts[1] + pad_amounts_odd[1]),
                                (pad_amounts[2], pad_amounts[2] + pad_amounts_odd[2])),
                    mode='constant', constant_values=0)

print("padded_result : ", padded_result)
print("padded_result : ", padded_result.shape)
padded_result = padded_result.astype(np.uint8)

resample_padded = np.pad(ensemble_result, ((pad_size_z, pad_size_z), (pad_size_y, pad_size_y), (pad_size_x, pad_size_x)), mode='constant')

result = sitk.GetImageFromArray(ensemble_result)
result.CopyInformation(gt_mask_sitk)

result.SetOrigin(origin)
result.SetSpacing(original_spacing)
result.SetDirection(direction)
sitk.WriteImage(result, '/mai_nas/LSH/4class_val_save/'+data_name[:-4]+'.nii.gz')


# 예시 numpy 배열과 spacing 정보
example_array = np.random.rand(100, 100, 100)
spacing = (1.0, 1.0, 1.0)

# NIfTI 파일로 저장할 경로
output_file_path = "output.nii.gz"

# Numpy 배열을 NIfTI 파일로 저장
save_numpy_as_nifti(padded_result, original_spacing, save_dir+data_name[:-4]+'.nii.gz')

print("NIfTI file saved.")




out_size = [
int(np.round(whole_crop_data[0] * (prepro_spacing[0] / original_spacing[0]))),
int(np.round(whole_crop_data[1] * (prepro_spacing[1] / original_spacing[1]))),
int(np.round(whole_crop_data[2] * (prepro_spacing[2] / original_spacing[2])))]

resampler = sitk.ResampleImageFilter()

resampler.SetOutputSpacing(original_spacing)
resampler.SetSize(out_size)
resampler.SetOutputDirection(direction)
resampler.SetOutputOrigin(origin)
resampler.SetDefaultPixelValue(ori_test_data.GetPixelIDValue())
resampler.SetInterpolator(sitk.sitkBSpline)
resampled_image = resampler.Execute(ensemble_result)


## 다시 array로 돌려서 패딩 후 GetImageFromArray하는 게 맞을지?

resampled_image_size = resampled_image.GetSize()    ##(z, , )
resample_origin = resampled_image.GetOrigin()
resample_direction = resampled_image.GetDirection()




resampled_image = sitk.GetArrayFromImage(resampled_image)


pad_size_z = int(np.ceil((original_size[0] - resampled_image_size[0])/2))
pad_size_y = int(np.ceil((original_size[1] - resampled_image_size[1])/2))
pad_size_x = int(np.ceil((original_size[2] - resampled_image_size[2])/2))


resample_padded = np.pad(resampled_image, ((pad_size_z, pad_size_z), (pad_size_y, pad_size_y), (pad_size_x, pad_size_x)), mode='constant')


resample_padded = sitk.GetImageFromArray(resample_padded)

# resample_padded.SetOrigin(origin)
resample_padded.SetSpacing(original_spacing)
# resample_padded.SetDirection(direction)

sitk.WriteImage(resample_padded, save_dir+i[:-4]+'.nii.gz')