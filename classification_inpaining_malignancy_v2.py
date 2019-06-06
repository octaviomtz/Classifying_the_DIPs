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
import random

if torch.cuda.is_available():
    device = 'cuda'

version = 2
path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification/'
path_inpain = f'{path_data}inpainted/'
path_original = f'{path_data}original/'
df= pd.read_csv(f'{path_data}df_classify_inpainted_malignancy.csv')
df_results_all = pd.DataFrame()

train_loss_folds, val_loss_folds, train_accuracy_folds, val_accuracy_folds = [], [], [], []  
folds = 5
X = df['names'].values
y = df['malignancy'].values
idx_all = np.arange(0,len(X)) # shuffle all indices
random.shuffle(idx_all)
for fold in range(folds):
    # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    idx_train, idx_val, idx_test = np.split(idx_all, [int(.8 * len(idx_all)), int(.9 * len(idx_all))])
    X_train = X[idx_train]; X_val = X[idx_val]; X_test = X[idx_test]
    y_train = y[idx_train]; y_val = y[idx_val]; y_test = y[idx_test]
    # roll the indices array 
    idx_all = np.roll(idx_all, len(idx_test)+len(idx_val))

    # Class imbalance, dataset and dataloader
    sampler = class_imbalance_sampler(y_train-1)
    dataset_train = Dataset_malignacy(X_train, y_train, path_original, transform=True)
    dataset_test = Dataset_malignacy(X_test, y_test, path_original, transform=False)
    dataset_val = Dataset_malignacy(X_val, y_val, path_original, transform=False)
    dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler)
    dataloader_test = DataLoader(dataset_test, batch_size=32)
    dataloader_val = DataLoader(dataset_val, batch_size=32)

    # CNN model
    model = ResNet18()
    # model = Darknet([2,2,2],2)
    model_name = f'resnet18_v{version}'
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
    figure_name = f'{model_name}_lr_{lr_str}'

    epochs = 50
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    debugging = []
    best_loss_val = 100
    # final test variables
    loss_test_epoch = 0
    total_test = 0
    all_pred_proba = []
    prediction_test_best_model = []
    y_test_all = []
    correct_test = 0

    for epoch in tqdm(range(epochs), total = len(range(epochs)), desc=f'training fold = {fold}'):
        epoch_loss = 0
        epoch_loss_val = 0
        loss_train_epoch = 0
        correct_train = 0
        total_train = 0
        loss_val_epoch = 0
        prediction_all = []
        y_val_all = []
        correct_val = 0
        total_val = 0
        
        model.train()
        for idx, (x_train, y_train) in enumerate(dataloader_train):

            start = time.time()
            pred, loss, batch_total, batch_correct, _ = loss_batch(model, criterion, x_train, y_train, device)
            correct_train += batch_correct
            total_train += batch_total
            loss_train_epoch += loss.detach().cpu().numpy()
            
            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        accuracy_train = 100 * correct_train / total_train
        
        model.eval() 
        # Val set       
        with torch.no_grad():
            for idx_val, (x_val, y_val) in enumerate(dataloader_val):
                pred_val, loss_val_batch, batch_total_val, batch_correct_val, _ = loss_batch(model, criterion, x_val, y_val, device)

                loss_val_epoch += loss_val_batch.detach().cpu().numpy()
                total_val += batch_total_val   
                correct_val += batch_correct_val
                
            accuracy_val = 100 * correct_val/ total_val

            # Save best model according to test set
            model_saved_name = f'results/CNN_models/{figure_name}'
            if best_loss_val > loss_val_epoch and epoch >= 2:
                best_loss_val = loss_val_epoch
                best_val_acc = accuracy_val
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),'optim_dict' : opt.state_dict()}, model_saved_name)
            
        train_loss.append(loss_train_epoch)
        val_loss.append(loss_val_epoch)
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)

    # Test on the final set
    checkpoint = torch.load(model_saved_name)
    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optim_dict'])
    os.remove(model_saved_name)

    # Final set
    model.eval()
    with torch.no_grad():
        for idx_test, (x_test, y_test) in enumerate(dataloader_test):
            pred_test, loss_test_batch, batch_total_test, batch_correct_test, predicted_test = loss_batch(model, criterion, x_test, y_test, device)
            
            loss_test_epoch += loss_test_batch.detach().cpu().numpy()
            total_test += batch_total_test
            correct_test += batch_correct_test
            
            # save all predictions
            all_pred_proba.extend(pred_test.detach().cpu().numpy())
            prediction_test_best_model.extend(predicted_test.detach().cpu().numpy())
            y_test_all.extend(y_test.detach().cpu().numpy())

        accuracy_test = 100 * correct_test / total_test

    del model

    train_loss_folds.append(train_loss)
    val_loss_folds.append(val_loss)
    train_accuracy_folds.append(train_accuracy)
    val_accuracy_folds.append(val_accuracy)

    figure_name_loss='loss_'+figure_name
    figure_name_acc='acc_'+figure_name

    df_results = pd.DataFrame.from_dict({f'train_loss_{fold}':train_loss,
    f'val_loss_{fold}':val_loss,f'train_accuracy_{fold}':train_accuracy,
    f'val_accuracy_{fold}':val_accuracy, f'all_pred_proba_{fold}':all_pred_proba,
    f'best_val_loss': [best_loss_val],  f'best_val_acc': [best_val_acc],
    f'prediction_test_set_{fold}':prediction_test_best_model, 
    f'y_test_set_{fold}':y_test_all, f'accuracy_test_{fold}':[accuracy_test]}, 
    orient='index')
    df_results = df_results.transpose()
    df_results_all = pd.concat([df_results_all, df_results], axis=1)

df_results_all.to_csv(f'results/results_{figure_name}.csv', index=False)
save_train_test_plot_folds(train_loss_folds, val_loss_folds, 'Loss', figure_name_loss)
save_train_test_plot_folds(train_accuracy_folds, val_accuracy_folds, 'Acc', figure_name_acc)