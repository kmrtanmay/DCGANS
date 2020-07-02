import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

def get_traindata(transform,batch_size):

	#Loading Training DataSets  
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

	return trainloader


def get_testdata(transform,batch_size):
	
    #Loading Test DataSets 
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
	return testloader