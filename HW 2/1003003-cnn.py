#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:34:54 2020

@author: nailinzhao
"""

import numpy 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt
 
labels = {
        0 : "T-shirt",
        1 : "Trouser",
        2 : "Pullover",
        3 : "Dress",
        4 : "Coat",
        5 : "Sandal",
        6 : "Shirt",
        7 : "Sneaker",
        8 : "Bag",
        9 : "Ankle boot"
        }
#Q3.1
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # initialize layers here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=0)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=1)
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)
 
    def forward(self, x):
        # invoke the layers here
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
     
# Q3.2
#Complete the main function
        
def main():
    N_EPOCH = 20
    L_RATE = 0.001
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST('../data', train=False, download=True, transform=transforms.ToTensor())
 
    ##### Use dataloader to load the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10,shuffle=True, num_workers=4)
 
 
    model = CNN().to(device)
    #criterion = nn.MSELoss()
 
    optimizer = optim.SGD(model.parameters(), lr=L_RATE)
 
    test(model, device, test_loader, True)
    for epoch in range(1, N_EPOCH + 1):
        test(model, device, test_loader, False)
        train(model, device, train_loader, optimizer, epoch)
 
    test(model, device, test_loader, True)
 
 
# Q3.3
     
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target)in enumerate (train_loader):
        data, target = data.to(device), target.to(device)
         
        # Fill in here
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        if batch_idx % 100 == 0:
            print("Epoch:", epoch, ", loss:", loss.item())
             
 
#Q3.4
             
def test(model, device, test_loader, enable_imshow):
    model.eval()
    correct = 0
    exampleSet = False
    example_data = numpy.zeros([10, 28, 28])
    example_pred = numpy.zeros(10)
 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
 
            # fill in here
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == target).sum().item()
            
            if not exampleSet and enable_imshow:
                for i in range(10):
                    example_data[i] = data[i][0].to("cpu").numpy()
                    example_pred[i] = pred[i].to("cpu").numpy()
 
                exampleSet = True
 
    print("Test set accuracy: ", 100. * correct / len(test_loader.dataset), "%")
         
    for i in range(10):
             
        plt.subplot(2, 5, i + 1)
             
        plt.imshow(example_data[i], cmap="gray", interpolation="none")
             
        plt.title(labels[example_pred[i]])
             
        plt.xticks([])
             
        plt.yticks([])
    plt.show()


main()