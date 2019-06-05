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

from torch.autograd import Variable
from torch import optim

from resnet import *
from darknet import *
from utils_classify_DIPs import *
from sklearn.model_selection import KFold
from decimal import Decimal

if torch.cuda.is_available():
    device = 'cuda'

# ## Data
path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification/'
path_inpain = f'{path_data}inpainted/'
path_original = f'{path_data}original/'

df= pd.read_csv(f'{path_data}df_classify_inpainted_malignancy.csv')

X = df['names'].values
y = df['malignancy'].values
kf = KFold(n_splits=10, shuffle=True)
train_loss_folds, test_loss_folds, train_accuracy_folds, test_accuracy_folds = [], [], [], []  
for fold, (idx_train, idx_test) in enumerate(kf.split(X, y)):
    if fold==5:break # Only do two of the ten folds (so each training set has 90%)
    X_train= X[idx_train]
    X_test= X[idx_test]
    y_train= y[idx_train]
    y_test=y[idx_test]

    # Class imbalance, dataset and dataloader
    sampler = class_imbalance_sampler(y_train-1)
    dataset_train = Dataset_malignacy(X_train, y_train, path_original, transform=True)
    dataset_test = Dataset_malignacy(X_test, y_test, path_original, transform=False)
    dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler)
    dataloader_test = DataLoader(dataset_test, batch_size=32)

    # CNN model
    model = ResNet18()
    # model = Darknet([2,2,2],2)
    model.apply(weights_init)

    if fold == 0:
        print('finding LR')
        lr=1e-7
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = F.cross_entropy
        lr_finder_lr, lr_finder_loss = find_lr_get_losses(model, opt, criterion, dataloader_train)
        lr, LR_idx, lr_finder_loss_filtered, idx_neg_slp  = find_lr_get_lr(lr_finder_lr, lr_finder_loss)
        #opt = optim.Adam(model.parameters(), lr=LR)

    if torch.cuda.is_available():
        model.cuda(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy

    #figure_name
    lr_str = f'{Decimal(lr):.2E}'
    lr_str = lr_str.replace('.','_')
    #fold = '5folds'
    figure_name = f'resnet18_lr={lr_str}_{fold}'

    epochs = 40
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    debugging = []
    for epoch in tqdm(range(epochs), total = len(range(epochs)), desc=f'training fold = {fold}'):
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
            
        train_loss.append(loss_train_epoch)
        test_loss.append(loss_test_epoch)
        train_accuracy.append(accuracy_train)
        test_accuracy.append(accuracy)

    #save_train_test_plot(train_loss, test_loss, 'Loss', figure_name)

    train_loss_folds.append(train_loss)
    test_loss_folds.append(test_loss)
    train_accuracy_folds.append(train_accuracy)
    test_accuracy_folds.append(test_accuracy)

figure_name_loss='loss_'+figure_name
figure_name_acc='acc_'+figure_name
df_results = pd.DataFrame.from_dict({'train_loss':train_loss,'test_loss':test_loss,'train_accuracy':train_accuracy,'test_accuracy':test_accuracy})
df_results.to_csv(f'results/{figure_name}')
save_train_test_plot_folds(train_loss_folds, test_loss_folds, 'Loss', figure_name_loss)
save_train_test_plot_folds(train_accuracy_folds, test_accuracy_folds, 'Acc', figure_name_acc)