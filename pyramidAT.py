# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:00:46 2023

@author: Daeha Kim
@email: kdhht5022@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as nnf

import cv2
import numpy
import matplotlib.pyplot as plt

global INTERP_MODE
INTERP_MODE = 'nearest'  #or 'bicubic'


def get_perturbed_image_simple(images, delta, mode='nearest'):
    return images + sum(M[i]*nnf.interpolate(delta[i].permute(2,0,1).unsqueeze(0), size=(H,H), mode=mode) for i in delta)
    
    
def get_perturbed_image_iter(images, delta, mode='nearest'):
    for i in range(len(S)):
        residue = M[i]*nnf.interpolate(delta[i].permute(2,0,1).unsqueeze(0), size=(H,H), mode=mode)  # [1, 3, 224, 224]
        plt.imshow(delta[i].detach().numpy()*255.)
        plt.xlabel('delta'); plt.axis('off'); plt.show()
        plt.imshow(residue[0].permute(1,2,0).detach().numpy()*255.)
        plt.xlabel('gradients with level {}'.format(i)); plt.axis('off'); plt.show()
        images += residue
    return images


def get_perturbed_loss_and_grad(images, labels, delta):
    
    ddict, gdict = {}, {}
    for i in range(len(S)):
        ddelta = Variable(delta[i], requires_grad=True)
        ddelta.requires_grad = True
        ddict.update({i: ddelta})

    pred = model( get_perturbed_image_simple(images.unsqueeze(0), ddict, mode=INTERP_MODE) )
    loss = F.nll_loss(pred, labels)
    loss.sum().backward(retain_graph=True)
    
    for i in range(len(S)):
        gdict.update({i: ddict[i].grad})
    return gdict


def pyramidAT(images, model, mode=INTERP_MODE, n_steps=10):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    
    if isinstance(images, torch.Tensor):
        images = Variable(images, requires_grad=True)
    elif isinstance(images, numpy.ndarray):
        images = transform(images)
        images = Variable(images, requires_grad=True)
        
    labels = torch.topk(model(images.unsqueeze(0)), k=1)[1][0]  # (pseudo) label from pre-trained model
    
    delta = {i: torch.zeros((int(H/s),int(H/s),3)) for (i,s) in enumerate(S)}
    for _ in range(n_steps):
        delta = {i: delta[i] + lr * torch.sign(get_perturbed_loss_and_grad(images, labels, delta)[i]) for i in delta}
        
    perturbed_image = torch.clip(get_perturbed_image_simple(images.unsqueeze(0), delta, mode=INTERP_MODE), BOUNDS[0], BOUNDS[1])
    return perturbed_image


if __name__ == "__main__":
    
    lr          = 3./255
    H           = 224
    M           = [20,10,1]
    S           = [32,16,1]
    BOUNDS      = [0,1]
    n_steps     = 10
    
    model = models.resnet50(pretrained='imagenet').eval()
    images = cv2.resize(cv2.imread('imgs/golf_ball.jfif'), (224,224))
    
    perturbed_image = pyramidAT(images, model, mode=INTERP_MODE, n_steps=n_steps)
    
    plt.imshow(images)
    plt.axis('off')
    plt.show()
    plt.imshow(cv2.cvtColor(perturbed_image[0].permute(1,2,0).detach().numpy(), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()