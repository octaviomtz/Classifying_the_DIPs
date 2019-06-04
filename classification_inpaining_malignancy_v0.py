import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchsummary import summary
import numpy as np
import pandas as pd
import os
import visdom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from copy import copy

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import Tensor
from torch import optim
from torch.utils.data import Sampler, WeightedRandomSampler

from resnet import *
from darknet import *

if torch.cuda.is_available():
    device = 'cuda'

colors = ['#222a20', '#e6c138', '#869336',  '#44472f', '#eef9c8']

def save_train_test_plot(train_, test_, acc_or_loss, title):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(train_, c=colors[0], label="train")
    ax.plot(test_, c=colors[1], label='test')
    # Remove the plot frame lines. 
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    ax.set_ylabel(acc_or_loss, fontsize=16)  
    ax.set_xlabel("Epochs", fontsize=16)  
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title(title)
    fig.savefig(f'results/figures/{title}')


# ## Data
path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification/'
path_inpain = f'{path_data}inpainted/'
path_original = f'{path_data}original/'

df= pd.read_csv(f'{path_data}df_classify_inpainted_malignancy.csv')

X = df['names'].values
y = df['malignancy'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
len(X_train), len(X_test), len(y_train), len(y_test)

type(y_train), np.shape(y_train)


# ## dataset & dataloader
# Class imbalance
train_imbalance = torch.from_numpy(np.asarray(y_train-1)*1)
# Get the weight of each class and assign the corresponding weight to each element to the
weight_class1 = len(train_imbalance)/sum(train_imbalance).item() 
weight_class0 = len(train_imbalance)/(len(train_imbalance)-sum(train_imbalance)).item()
weights_array = []
for i in train_imbalance:
    if i==1: weights_array.append(weight_class1)
    else: weights_array.append(weight_class0)        
sampler = WeightedRandomSampler(weights_array, len(weights_array))

class Dataset_malignacy(Dataset):
    def __init__(self, x_train, y_train, path_dataset, transform = False):
        self.X = [f'{i}.npy' for i in x_train]
        self.y = y_train-1
        self.path_dataset = path_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def rotate_axis(self, img_, axes):
        """Rotate around the three axes maximum three times per axis"""
        img_ = copy(img_)
        num_rot = 1#np.random.randint(1,4)
        img_ = np.rot90(img_, num_rot, axes)
        return img_

    def __getitem__(self, idx):
        img = np.load(f'{self.path_dataset}{self.X[idx]}')
        
        if self.transform:
            if np.random.rand() > 0.5:
                img = self.rotate_axis(img, (0,1))
            if np.random.rand() > 0.5:
                img = self.rotate_axis(img, (0,2))
            if np.random.rand() > 0.5:
                img = self.rotate_axis(img, (1,2))
                            
        target = self.y[idx]
        target = torch.from_numpy(np.expand_dims(target,-1)).float()
        target = Tensor(target).long().squeeze()
        
        img = Tensor(img.copy())
        return img, target

dataset_train = Dataset_malignacy(X_train, y_train, path_original, transform=True)
dataset_test = Dataset_malignacy(X_test, y_test, path_original, transform=False)

dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler)
dataloader_test = DataLoader(dataset_test, batch_size=32)

model = ResNet18()
# model = Darknet([2,2,2],2)
if torch.cuda.is_available():
    model.cuda(device)

lr=1e-4
opt = optim.Adam(model.parameters(), lr=lr)
criterion = F.cross_entropy
# criterion = nn.CrossEntropyLoss()

epochs = 5
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
for epoch in tqdm(range(epochs), desc='train batch', total = len(range(epochs))):
    if epoch==0:print(epoch)
    epoch_loss = 0
    epoch_loss_val = 0
    loss_train_epoch = 0
    correct_train = 0
    total_train = 0
    loss_test_epoch = 0
    prediction_all = []
    y_test_all = []

    for idx, (x_train, y_train) in enumerate(dataloader_train):

        start = time.time()
        model.train()
        
        if torch.cuda.is_available():
            x_train = Variable(x_train.cuda(device))
            y_train = Variable(y_train.cuda(device))
        else:
            x_train = Variable(x_train)
            y_train = Variable(y_train)
            
        pred = model(x_train)
        loss = criterion(pred, y_train)
        _, predicted_train = torch.max(pred.data, 1)
        total_train += y_train.size(0)
        loss_train_epoch += loss.detach().cpu().numpy()
        
        # Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Compute train accuracies
        if torch.cuda.is_available():
            correct_batch = (predicted_train.cpu() == y_train.cpu()).sum().item()
            correct_train += correct_batch
        else:
            correct_batch = (predicted_train == y_train).sum().item()
            correct_train += correct_batch
        
    accuracy_train = 100 * correct_train / total_train
    
    # Test set
    model.eval()
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for idx_test, (x_test, y_test) in enumerate(dataloader_test):
            if torch.cuda.is_available():
                x_test = Variable(x_test).cuda(device)
                y_test = Variable(y_test).cuda(device)
            
            pred_test = model(x_test)
            
            loss_test_batch = criterion(pred_test, y_test)
            loss_test_epoch += loss_test_batch.detach().cpu().numpy()
            
            _, predicted = torch.max(pred_test.data, 1)
            total_test += y_test.size(0)
            
            # save all predictions of last epoch
            if epoch == epochs-1:
                prediction_all.extend(predicted.detach().cpu().numpy())
                y_test_all.extend(y_test.detach().cpu().numpy())
                
            # Compute test accuracies
            if torch.cuda.is_available():
                correct_batch_test = (predicted.cpu() == y_test.cpu()).sum().item()
                correct_test += correct_batch_test
            else:
                correct_batch_test = (predicted == y_test).sum().item()
                correct_test += correct_batch_test
            
            accuracy = 100 * correct_test/ total_test
        
    train_loss.append(loss_train_epoch.copy())
    test_loss.append(loss_test_epoch)
    train_accuracy.append(accuracy_train)
    test_accuracy.append(accuracy)

    save_train_test_plot(train_loss, test_loss, 'Loss', 'resnet18')