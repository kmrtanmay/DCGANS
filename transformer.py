import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

def transformer(image_size):
    
    #Image Transformations
	transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return transform