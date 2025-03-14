#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:10:36 2021

@author: compu
"""


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import math
# import pytorch_ssim
# from ssim import ssim as ssimFast
import time
import io
from PIL import Image
import torchvision
import random
# from DISTS_pytorch import DISTS

from collections import defaultdict
import cv2
from PIL import Image
import copy

import torch.nn as nn

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#%%
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

def  read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        if (0x0028,0x3000) in dicom and dicom[(0x0028,0x3000)][0][(0x0028,0x3002)].value != None:
            data = apply_voi_lut(apply_modality_lut(dicom.pixel_array, dicom), dicom)
        else:
            data = dicom.pixel_array
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = (data - np.min(data)).astype('float32')
    data = data / np.max(data)
    data = (data * 255)
        
    return data

#%%
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def fftLoss(pred, label):
    predF = torch.abs(torch.fft.fftshift(torch.fft.fft2(pred)))
    labelF = torch.abs(torch.fft.fftshift(torch.fft.fft2(label)))
    return torch.mean((predF - labelF)**2)

def asymmMSELoss(pred, label):
    alpha = 0.3
    diff = pred - label
    return torch.mean((0.5 - torch.sign(diff)*(0.5-alpha)) * diff**2)

def TVLoss(pred):
    w_variance = torch.mean(torch.pow(pred[:,:,:,:-1] - pred[:,:,:,1:], 2))
    h_variance = torch.mean(torch.pow(pred[:,:,:-1,:] - pred[:,:,1:,:], 2))
    return w_variance + h_variance

def TVdiffLoss(pred, label):
    return torch.abs(TVLoss(pred)-TVLoss(label))

def contrastiveLoss(predA, predB, labelA, labelB):
    criterion_DISTS = DISTS().cuda()
    similarity_same = criterion_DISTS(predA, predB, require_grad=True, batch_average=True)
    similarity_diff1 = criterion_DISTS(predA, labelA, require_grad=True, batch_average=True)
    similarity_diff2 = criterion_DISTS(predA, labelB, require_grad=True, batch_average=True)
    similarity_diff3 = criterion_DISTS(predB, labelA, require_grad=True, batch_average=True)
    similarity_diff4 = criterion_DISTS(predB, labelB, require_grad=True, batch_average=True)
    return -similarity_same + torch.log(torch.exp(similarity_diff1)+torch.exp(similarity_diff2)+torch.exp(similarity_diff3)+torch.exp(similarity_diff4))

def active_contour_loss(pred, label):
    '''
    pred, label: tensor of shape (B,C,H,W), where label[:,:,region_in_contour]==1, label[:,:,region_out_contour]==0
    '''
    
    pred = pred.contiguous()
    label = label.contiguous()
    
    ## length term
    delta_r = pred[:,:-1,1:,:] - pred[:,:-1,:-1,:] # horizontal gradient (B, C, H-1, W)
    delta_c = pred[:,:-1,:,1:] - pred[:,:-1,:,:-1] # vertical gradient (B, C, H, W-1)
    
    delta_r = delta_r[:,:-1,1:,:-2] ** 2 # (B, C, H-2, W-2)
    delta_c = delta_c[:,:-1,:-2,1:] ** 2 # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)
    
    epsilon = 1e-8
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))
    
    ## region term
    c_in = torch.ones_like(pred[:,:-1,1:,:])
    c_out = torch.zeros_like(pred[:,:-1,1:,:])
    
    region_in = torch.mean(pred[:,:-1,1:,:] * (label[:,:-1,1:,:] - c_in)**2)
    region_out = torch.mean((1-pred[:,:-1,1:,:]) * (label[:,:-1,1:,:] - c_out)**2)
    region = region_in + region_out
    
    return lenth + region






def dice_score(pred, label):
    # pred = F.sigmoid(pred)
    
    smooth = 1
    
    pred = pred.contiguous().view(-1)
    label = label.contiguous().view(-1)
    
    intersection = (pred * label).sum()
    dice = (2*intersection + smooth) / (pred.sum() + label.sum() + smooth)
    
    return dice

def dice_loss(pred, label):
    dice = dice_score(pred[:,:-1,:,:], label[:,:-1,:,:])
    
    return 1 - dice

def weightedCE_loss(pred, label, weight):
    # BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = nn.BCELoss(reduction='none')
    
    loss = BCE_loss(pred[:,:-1,:,:], label[:,:-1,:,:]) * weight
    loss = loss.sum(dim=1)
    loss = loss.view(pred[:,:-1,:,:].shape[0],-1) / weight.view(pred[:,:-1,:,:].shape[0],-1)  #디버깅 진입 안됨?????
    
    return loss.mean()

def weight_map(mask, w0 = 10, sigma = 5, background_class = 1):
    
    # Fix mask datatype (should be unsigned 8 bit)
    if mask.dtype != 'uint8': 
        mask = mask.astype('uint8')
    
    # Weight values to balance classs frequencies
    wc = _class_weights(mask)
    
    # Assign a different label to each connected region of the image
    _, regions = cv2.connectedComponents(mask)
    
    # Get total no. of connected regions in the image and sort them excluding background
    region_ids = sorted(np.unique(regions))
    region_ids = [region_id for region_id in region_ids if region_id != background_class]
        
    if len(region_ids) > 1: # More than one connected regions

        # Initialise distance matrix (dimensions: H x W x no.regions)
        distances = np.zeros((mask.shape[0], mask.shape[1], len(region_ids)))

        # For each region
        for i, region_id in enumerate(region_ids):

            # Mask all pixels belonging to a different region
            m = (regions != region_id).astype(np.uint8)# * 255
        
            # Compute Euclidean distance for all pixels belongind to a different region
            distances[:, :, i] = cv2.distanceTransform(m, distanceType = cv2.DIST_L2, maskSize = 0)

        # Sort distances w.r.t region for every pixel
        distances = np.sort(distances, axis = 2)

        # Grab distance to the border of nearest region
        d1, d2 = distances[:, :, 0], distances[:, :, 1]

        # Compute RHS of weight map and mask background pixels
        w = w0 * np.exp(-1 / (2 * sigma ** 2)  * (d1 + d2) ** 2) * (regions == background_class)

    else: # Only a single region present in the image
        w = np.zeros_like(mask)

    # Instantiate a matrix to hold class weights
    wc_x = np.zeros_like(mask)
    
    # Compute class weights for each pixel class (background, etc.)
    for pixel_class, weight in wc.items():
    
        wc_x[mask == pixel_class] = weight
    
    # Add them to the weight map
    w = w + wc_x
    
    return w

def _class_weights(mask):
    ''' Create a dictionary containing the classes in a mask,
        and their corresponding weights to balance their occurence
    '''
    wc = defaultdict()

    # Grab classes and their corresponding counts
    unique, counts = np.unique(mask, return_counts = True)

    # Convert counts to frequencies
    counts = counts / np.product(mask.shape)

    # Get max. counts
    max_count = max(counts)

    for val, count in zip(unique, counts):
        wc[val] = max_count / count
    
    return wc
    

#%%
def Poisson(x, eta):
    return np.clip(np.random.poisson(x * eta) / eta, 0, 1)

def Gaussian(x, sigma):
    return np.clip(x + np.random.normal(0, sigma, x.shape), 0, 1)

def SaltnPepper(x, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(x.shape)
    thres = 1 - prob 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            rdn = random.random()
            rdnV = random.random()
            if rdn < prob:
                # output[i,j] = x[i,j] - (rdnV*0.2 - 0.1)
                output[i,j] = 0
            elif rdn > thres:
                # output[i,j] = x[i,j] + (rdnV*0.2 - 0.1)
                output[i,j] = 1
            else:
                output[i,j] = x[i,j]
    return np.clip(output, 0, 1)

def SaltnPepperSoft(x, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(x.shape)
    thres = 1 - prob 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            rdn = random.random()
            rdnV = random.random()
            if rdn < prob:
                # output[i,j] = x[i,j] - (rdnV*0.2 - 0.1)
                output[i,j] = x[i,j] * (random.random()*0.1+0.8)
            elif rdn > thres:
                # output[i,j] = x[i,j] + (rdnV*0.2 - 0.1)
                output[i,j] = x[i,j] / (random.random()*0.1+0.8)
            else:
                output[i,j] = x[i,j]
    return np.clip(output, 0, 1)

def elastic_transform(image, label, alpha=35, sigma=5, alpha_affine=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    imageT = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelT = copy.deepcopy(label)
    for i in range(label.shape[-1]-1):
        labelT[:,:,i] = cv2.warpAffine(label[:,:,i], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelT[labelT>0.5] = 1
    labelT[labelT<=0.5] = 0
    labelT[:,:,-1] = 1 - np.sum(labelT[:,:,:-1],axis=-1)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # dz = np.zeros_like(dx)

    # x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    imageT = map_coordinates(imageT, indices, order=1, mode='reflect').reshape(shape)
    for i in range(label.shape[-1]):
        labelT[:,:,i] = map_coordinates(labelT[:,:,i], indices, order=1, mode='reflect').reshape(shape)
    
    return imageT, labelT


def elastic_transform2D(image, label, alpha=35, sigma=5, alpha_affine=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    imageT = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelT = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    imageT = map_coordinates(imageT, indices, order=1, mode='reflect').reshape(shape)
    labelT = map_coordinates(labelT, indices, order=1, mode='reflect').reshape(shape)

    return imageT, labelT

def calculate_psnr(img, label):
    # img = img / torch.max(label)
    # label = label / torch.max(label)
    maxI = 1
    mse = torch.mean((img - label)**2)
#    if torch.max(mse) == 0:
#        return float('inf')
    return 20 * torch.log10(maxI / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    # img1 = img1 / torch.max(img2)
    # img2 = img2 / torch.max(img2)
    SSIM = 0
    for i in range(img1.shape[0]):
        SSIM += pytorch_ssim.ssim(img1[i,:,:,:][None,:,:,:], img2[i,:,:,:][None,:,:,:])
    SSIM = SSIM / img1.shape[0]
    return SSIM
    # return ssimFast(img1, img2)


#%%
def save_images(val_data, epoch, opt, save_directory):
    '''
    val_data[0] : input
    val_data[1] : output
    val_data[2] : label
    '''
    n = 0
    DICE = dice_score(val_data[1],val_data[2])
    
    titles = ['%03d: Input' % epoch,
              'Output (DICE %.4f)' % DICE,
              'Label']
    
    fig, axs = plt.subplots(2,8)
    
    axs[0,0].imshow(val_data[0][n,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[0,0].set_title('%03d: Input' % epoch, fontdict={'fontsize':3})
    axs[0,0].axis('off')
    
    axs[1,0].imshow(np.zeros((256,256)), cmap='gray', vmin=0, vmax=1)
    axs[1,0].axis('off')
    
    for i in range(val_data[-1].shape[1]-1):
        axs[0,i+1].imshow(val_data[1][n,i,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[0,i+1].axis('off')
        
        axs[1,i+1].imshow(val_data[2][n,i,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1,i+1].axis('off')
    
    axs[0,7].imshow(1-val_data[1][n,-1,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0,7].set_title('Output (DICE %.4f)' % DICE, fontdict={'fontsize':3})
    axs[0,7].axis('off')
    axs[1,7].imshow(1-val_data[2][n,-1,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1,7].set_title('Label', fontdict={'fontsize':3})
    axs[1,7].axis('off')
    
    input = val_data[0][n,0,:,:].detach().cpu().numpy()
    output = val_data[1][n,:,:,:].detach().cpu().numpy()
    label = val_data[2][n,:,:,:].detach().cpu().numpy()
    
    sio.savemat(save_directory+"/valid_%03d.mat" % epoch, mdict={'input':input, 'output':output, 'label':label})
    
    plt.tight_layout()
    fig.savefig(save_directory+"/%d.png" % epoch, dpi = 300)
    plt.close()
