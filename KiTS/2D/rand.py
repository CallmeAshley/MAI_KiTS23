import numpy as np
import os
import random

data_dir = '/mai_nas/Benchmark_Dataset/MICCAI2023_KiTS/kits23/dataset'

patient_name = os.listdir(data_dir)

patient_name.sort()


random_elements = random.sample(patient_name, k=49)

random_elements.sort()

print(random_elements)
