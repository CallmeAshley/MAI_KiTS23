import SimpleITK as sitk
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

import datetime

dt_now = datetime.datetime.now()

dir  = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23/dataset/case_00000/segmentation.nii.gz'

seg = sitk.ReadImage(dir)
seg = sitk.GetArrayFromImage(seg).astype(np.int64)


transform = transforms.ToTensor()

seg = transform(seg)

seg = F.one_hot(seg, num_classes=4)




print('ff')
