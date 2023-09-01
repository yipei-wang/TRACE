import os
import time
import torch
import random
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models
from typing import Type, Any, Callable, Union, List, Optional
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from collections import deque


IMAGE_SIZE=224


def denorm(x, device = None):
    
    if x.shape[-3] == 3:
        if device == None:
            device = x.device
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx[:, 0, :, :] = xx[:, 0, :, :] * std[0] + mean[0]
        xx[:, 1, :, :] = xx[:, 1, :, :] * std[1] + mean[1]
        xx[:, 2, :, :] = xx[:, 2, :, :] * std[2] + mean[2]
    else:
        if device == None:
            device = x.device
        mean = 0.1307
        std = 0.3081
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx = xx * std + mean
        
    return xx
    
    
def normalize(x):
    if x.shape[-3] == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if len(x.shape) == 4:
            xx = x.clone().detach().to(x.device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(x.device)

        xx[:, 0, :, :] = (xx[:, 0, :, :] - mean[0]) / std[0]
        xx[:, 1, :, :] = (xx[:, 1, :, :] - mean[1]) / std[1]
        xx[:, 2, :, :] = (xx[:, 2, :, :] - mean[2]) / std[2]
    else:
        mean = 0.1307
        std = 0.3081
        if len(x.shape) == 4:
            xx = x.clone().detach().to(x.device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(x.device)

        xx = (xx - mean) / std
        
    return xx.squeeze()


    
def plot_tensor_image(image, save=None):
    toshow = image.detach().clone().cpu()
    toshow = toshow.squeeze()
    if toshow.max() > 1:
        toshow = denorm(toshow).squeeze()
    
    if len(toshow.shape) == 4:
        print("Plotting the entire batch")
        toshow = torchvision.utils.make_grid(toshow,nrow=10)
        figsize = (20,20)
    else:
        print("Plotting the single image")
        figsize = (5,5)
    
    plt.figure(figsize=figsize)
    plt.imshow(toshow.numpy().transpose((1,2,0)))
    plt.axis('off')
    if save is not None:
        print(f"save at {save}")
        plt.savefig(save,bbox_inches='tight')
    plt.show()
    

def get_n_params(model):
    n_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_param += torch.tensor(param.shape).prod()

    print(f'Then number of parameters is {n_param}.')
    
    



def masking(image, reference, idx = 0, edge = 7):
    '''
    image (torch.Tensor): 3 x IMAGE_SIZE x IMAGE_SIZE
    reference (str): the reference types
    '''
    channel = image.shape[-3]
    grid = edge**2
    patch_size = IMAGE_SIZE//edge
    
    assert idx >= 0 and idx <grid, 'idx out of range'
    x = idx //edge
    y = idx % edge
    masked = image.detach().clone()
    if reference == 'zero':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = 0
    elif reference == 'random':
        reference_values = normalize(torch.rand(channel,patch_size, patch_size).to(image.device)).squeeze()
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = reference_values
    elif reference == 'mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = image.mean()
    elif reference == 'patch_mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
        image[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size].mean()
    elif reference == 'blur':
        blurrer = transforms.GaussianBlur(kernel_size=(31, 31), sigma=(56, 56))
        blurred_img = blurrer(image)
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
            blurred_img[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size]
        
    return masked


def show_traj(image, traj, options, figsize = (15, 15), save=None):
    channel = image.shape[-3]
    masked = image.detach().clone().view(channel,IMAGE_SIZE,IMAGE_SIZE)
    toshow = [masked[None]]
    for ids in traj:
        masked = masking(masked, options.reference, ids, edge = options.edge)
        toshow.append(masked[None])
    toshow = torch.cat(toshow)
    plot_tensor_image(toshow,save=save)

    
def deletion_attribution(model, image, label, saliency, options, mode):
    saliency_superpatch = F.interpolate(
        torch.FloatTensor(saliency).view(1,1,saliency.shape[-1],saliency.shape[-1]),
        size = (options.edge, options.edge), mode = 'area')
    traj = saliency_to_traj(saliency_superpatch, mode)
    return traj_masking(model, image, label, traj, options.reference, options.edge)[0]

## Demnostration functions

def attribute(model, image, label, follow_prob = True, mode = 'top', reference = 'zero', grid = 64, insertion = False):
    image_size = image.shape[-1]
    if not insertion:
        traj, Pred, Prob = greedy(model, image, label, follow_prob, mode, reference)
        saliency = torch.zeros(1, grid).to(image.device)
        for i in range(grid):
            if follow_prob:
                saliency[0, traj[i]] = -Prob[i+1] + Prob[i]
            else:
                saliency[0, traj[i]] = -Pred[i+1] + Pred[i]
        saliency = saliency.view(1,1,edge,edge)
        heatmap = F.interpolate(saliency, size = (image_size, image_size), mode = 'bilinear', align_corners = True)
        return saliency, heatmap
    else:
        traj, Pred, Prob = greedy_insertion(model, image, label, follow_prob, mode, reference)
        saliency = torch.zeros(1, grid).to(image.device)
        for i in range(grid):
            if follow_prob:
                saliency[0, traj[i]] = -Prob[i+1] + Prob[i]
            else:
                saliency[0, traj[i]] = -Pred[i+1] + Pred[i]
        heatmap = F.interpolate(saliency, size = (image_size, image_size), mode = 'bilinear', align_corners = True)
        return saliency, heatmap
    
def saliency_to_traj(saliency, criterion = 'MoRF'):
    if criterion == 'MoRF':
        return np.array(list(reversed(saliency.flatten().argsort()).detach().cpu().numpy()))
    elif criterion == 'LeRF':
        return np.array(list(saliency.flatten().argsort().detach().cpu().numpy()))
    
def traj_masking(model, image, label, traj, reference = 'zero', edge = 7):
    model.eval()
    masked = image.detach().clone()
    with torch.no_grad():
        inputs = [masked[None]]
        for ids in traj:
            masked = masking(masked, reference, ids, edge = edge)
            inputs.append(masked[None])
        inputs = torch.cat(inputs).to(image.device)
        prob = torch.softmax(model(inputs), dim = 1)
        return prob[:, label].detach().cpu().numpy(), inputs


def show_attr(image, traj, index = 3):
    attr = F.interpolate((torch.FloatTensor(np.array(traj).argsort() + 1).view(1,1,edge,edge)/grid)**index, 
                         size = (image_size, image_size), mode = 'bilinear', align_corners = False)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(denorm(image).detach().cpu().numpy().squeeze().transpose((1,2,0)))
    ax[0].axis('off')
    ax[1].imshow(attr.squeeze(), cmap = 'jet')
    ax[1].axis('off')
    ax[2].imshow(denorm(image).detach().cpu().numpy().squeeze().transpose((1,2,0)))
    ax[2].imshow(attr.squeeze(), cmap = 'jet', alpha = 0.3)
    ax[2].axis('off')
    plt.tight_layout()
    plt.show()

def func(model, image, label, traj, options):    
    if options.mode == 'Le-Mo':
        return traj_masking(model, image, label, traj, edge = options.edge, reference = options.reference)[0].sum() -\
    traj_masking(model, image, label, reversed(traj), edge = options.edge, reference = options.reference)[0].sum()
    
    elif options.mode == 'Mo':
        return traj_masking(model, image, label, traj, edge = options.edge, reference = options.reference)[0].sum()
    elif options.mode == 'Le':
        return -traj_masking(model, image, label, traj, edge = options.edge, reference = options.reference)[0].sum()
    else:
        print('WRONG MODE!')

def nC2_swap(sol, candidates):
    traj = sol.copy()
    positions = list(candidates[np.random.randint(0, len(candidates))])
    traj[positions] = np.flip(traj[positions])
    return traj



def kendall_distance(t1, t2):
    n = len(t1)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    ndisordered = np.logical_or(np.logical_and(t1[i] < t1[j], t2[i] > t2[j]), 
                                np.logical_and(t1[i] > t1[j], t2[i] < t2[j])).sum()
    return ndisordered/2