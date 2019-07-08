import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchsummary import summary
import numpy as np
import pandas as pd
import os
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
import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
seed=0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='select model to train the classifier')
parser.add_argument('lr', type=str, nargs='?')
parser.add_argument('data_epoch', type=str, nargs='?')
parser.add_argument('data_patched_generated', type=str, nargs='?')
args = parser.parse_args()
print(f'analyzing {args.data_epoch}, {args.data_patched_generated},  with {args.model} and lr = {args.lr}')
torch.cuda.set_device(0)
if torch.cuda.is_available():
    device = 'cuda'

version = 3

np.random.seed(seed)
path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification 2D - for workshop/'
path_inpain = f'{path_data}inpainted/'
path_original = f'{path_data}original/'
path_mask = f'{path_data}mask/'
#df= pd.read_csv(f'{path_data}df_classify_inpainted_malignancy.csv')
#df= pd.read_csv(f'{path_data}df_3agree.csv')
df_results_all = pd.DataFrame()

df3 = pd.read_csv('/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification 2D - for workshop/df_3agree.csv')
df3_names = df3.names.values
#print(df3.shape, 'df3')
# We put in the test sets those samples passed QC and that agreement in 2 or 3 reviewers
# path_qualit_assess = '/home/om18/Documents/KCL/Papers/miccai 2019 lung nodule deep image prior/code for figures/inpain_visual_eval/'
# df1 = pd.read_csv(path_qualit_assess + 'inpain_visual_eval_part1.txt', header=None)
# df2 = pd.read_csv(path_qualit_assess + 'inpain_visual_eval_part2.txt', header=None)
# df_qualit_assess = pd.concat([df1, df2])
# df_qualit_assess[0] = df_qualit_assess[0].apply(lambda x: x.split('.jpg')[0])
# df_qualit_assess_names = df_qualit_assess[0].values
#print(df_qualit_assess_names[0], 'df_qualit_assess_names')
#print(df3_names[0], 'df3_names')

## The cycleGAN generated 105 files, those files MUST be in the test sets:
# Get the cycleGAN files
names_cyclegan = os.listdir('/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification 2D - for workshop/cycleGANgenerated/last/generated/')
names_cyclegan = [i.split('.')[0] for i in names_cyclegan]
# Use the same name convention as previous scripts
df3_qc= names_cyclegan
df3_notqc= [i for i in df3_names if i not in names_cyclegan]


test_fold_n = 35 # we changed from 40 to 35 because we only have 105 samples
np.random.seed(seed)
df3_qc_names = np.random.permutation(df3_qc)
np.random.seed(seed)
df3_notqc_names = np.random.permutation(df3_notqc)
df3_train_all, df3_test40_all, df3_val40_all = [], [], []

for fold, i in enumerate(range(3)):
    df3_40test = df3_qc_names[:test_fold_n]
    df3_40val = df3_qc_names[test_fold_n:test_fold_n*2]
    df3_40train = df3_qc_names[test_fold_n*2:] # the rest put them back in the training set
    
    df3_train_names = list(df3_notqc_names) + list(df3_40train)
    
    df_test_fold = df3.loc[df3['names'].isin(df3_40test)]
    df_val_fold = df3.loc[df3['names'].isin(df3_40val)]
    df_train_fold = df3.loc[df3['names'].isin(df3_train_names)]
    
    df3_test40_all.append(df_test_fold)
    df3_val40_all.append(df_val_fold)
    df3_train_all.append(df_train_fold)
     # rotate the vector
    df3_qc_names = np.roll(df3_qc_names, test_fold_n)

train_loss_folds, val_loss_folds, train_accuracy_folds, val_accuracy_folds = [], [], [], []  
val_accuracy_best_folds = []

path_cycleGAN = '/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification 2D - for workshop/cycleGANgenerated/'
path_cycleGAN_used = f'{path_cycleGAN}{args.data_epoch}/{args.data_patched_generated}/'
print(path_cycleGAN_used)

for fold,i in enumerate(range(3)):
    df3_train = df3_train_all[i]
    df3_val40 = df3_val40_all[i]
    df3_test40 = df3_test40_all[i]
    X_train = df3_train['names'].values
    y_train = df3_train['malignancy'].values
    X_val = df3_val40['names'].values
    y_val = df3_val40['malignancy'].values
    X_test = df3_test40['names'].values
    y_test = df3_test40['malignancy'].values
    #print(np.shape(X_train), np.shape(X_val), np.shape(X_test), 'here')

    

    sampler = class_imbalance_sampler(y_train-1)
    #dataset_train = Dataset_malignacy2D_3sets(X_train, y_train, path_to_use, transform=True)
    dataset_train = Dataset_malignacy2D_3sets(X_train, y_train, path_original, transform=True) #path_original
    dataset_test = Dataset_malignacy2D_3sets(X_test, y_test, path_cycleGAN_used, transform=False)
    dataset_val = Dataset_malignacy2D_3sets(X_val, y_val, path_cycleGAN_used, transform=False)
    dataloader_train = DataLoader(dataset_train, batch_size=32, sampler=sampler)
    dataloader_test = DataLoader(dataset_test, batch_size=32)
    dataloader_val = DataLoader(dataset_val, batch_size=32)
    data_name = path_cycleGAN_used.split('workshop/')[-1].replace('/','_')
    print(f'data_name = {data_name}')

    # CNN model
    if args.model == 'dark222':
        torch.manual_seed(seed)
        model = Darknet([2,2,2],2)
        cnn_model_name = args.model
    elif args.model == 'resnet50':
        torch.manual_seed(seed)
        model = ResNet50_2D()
        cnn_model_name = 'resnet50'
    else: 
        torch.manual_seed(seed)
        model = ResNet18_2D()
        cnn_model_name = 'resnet18'
    
    model_name = f'v{version}_{data_name}_{cnn_model_name}'
    torch.manual_seed(seed)
    model.apply(weights_init)

    if fold == 0 and args.lr != 'fixed':
        print('finding LR')
        lr=1e-7
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = F.cross_entropy
        lr_finder_lr, lr_finder_loss = find_lr_get_losses(model, opt, criterion, dataloader_train)
        lr, LR_idx, lr_finder_loss_filtered, idx_neg_slp  = find_lr_get_lr(lr_finder_lr, lr_finder_loss)
        #opt = optim.Adam(model.parameters(), lr=LR)

    if torch.cuda.is_available():
        model.cuda()
    if args.lr == 'fixed':
        lr = 1e-3
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy 

    #figure_name
    lr_str = f'{Decimal(lr):.2E}'
    lr_str = lr_str.replace('.','_')
    #fold = '5folds'
    figure_name = f'{model_name}_lr_{lr_str}'

    epochs = 200
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    debugging = []
    best_loss_val = 100
    # final test variables
    loss_test_epoch = 0
    total_test = 0
    pred_test_proba_all = []
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
        y_train_all = []
        correct_val = 0
        total_val = 0
        pred_val_proba_all = []
        pred_train_proba_all = []

        model.train()
        for idx, (x_train, y_train) in enumerate(dataloader_train):

            start = time.time()
            pred_train_proba, loss, batch_total, batch_correct, _ = loss_batch(model, criterion, x_train, y_train-1, device)
            correct_train += batch_correct
            total_train += batch_total
            loss_train_epoch += loss.detach().cpu().numpy()
            y_train_all.extend(y_train.detach().cpu().numpy())
            pred_train_proba_all.extend(pred_train_proba)
            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        accuracy_train = 100 * correct_train / total_train
        
        model.eval() 
        # Val set       
        with torch.no_grad():
            for idx_val, (x_val, y_val) in enumerate(dataloader_val):
                pred_val_proba, loss_val_batch, batch_total_val, batch_correct_val, _ = loss_batch(model, criterion, x_val, y_val-1, device)

                loss_val_epoch += loss_val_batch.detach().cpu().numpy()
                total_val += batch_total_val   
                correct_val += batch_correct_val
                pred_val_proba_all.extend(pred_val_proba)
                y_val_all.extend(y_val.detach().cpu().numpy())

            accuracy_val = 100 * correct_val/ total_val

            # Save best model according to val set
            model_saved_name = f'results/CNN_models/{figure_name}'
            if best_loss_val > loss_val_epoch and epoch >= 2:
                best_loss_val = loss_val_epoch
                best_val_acc = accuracy_val
                pred_val_proba_best = pred_val_proba_all
                y_val_best = y_val_all    
                y_train_best = y_train_all    
                pred_train_proba_best = pred_train_proba_all
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
            pred_test_proba, loss_test_batch, batch_total_test, batch_correct_test, predicted_test = loss_batch(model, criterion, x_test, y_test-1, device)
            
            loss_test_epoch += loss_test_batch.detach().cpu().numpy()
            total_test += batch_total_test
            correct_test += batch_correct_test
            
            # save all predictions
            pred_test_proba_all.extend(pred_test_proba)
            prediction_test_best_model.extend(predicted_test.detach().cpu().numpy())
            y_test_all.extend(y_test.detach().cpu().numpy())

        accuracy_test = 100 * correct_test / total_test

    del model

    train_loss_folds.append(train_loss)
    val_loss_folds.append(val_loss)
    train_accuracy_folds.append(train_accuracy)
    val_accuracy_folds.append(val_accuracy)
    val_accuracy_best_folds.append(best_val_acc)

    figure_name_loss='loss_'+figure_name
    figure_name_acc='acc_'+figure_name

    df_results = pd.DataFrame.from_dict({f'train_loss_{fold}':train_loss,
    f'val_loss_{fold}':val_loss,f'train_accuracy_{fold}':train_accuracy,
    f'val_accuracy_{fold}':val_accuracy,
    f'train_pred_proba{fold}':pred_train_proba_best, f'y_train_set_{fold}':y_train_best, 
    f'val_pred_proba{fold}':pred_val_proba_best, f'y_val_set_{fold}':y_val_best, 
    f'best_val_loss_{fold}': [best_loss_val],  f'best_val_acc_{fold}': [best_val_acc],
    f'test_pred_proba{fold}':pred_test_proba_all,
    f'prediction_test_set_{fold}':prediction_test_best_model, 
    f'y_test_set_{fold}':y_test_all, f'accuracy_test_{fold}':[accuracy_test]}, 
    orient='index')
    df_results = df_results.transpose()
    df_results_all = pd.concat([df_results_all, df_results], axis=1)
val_acc_best_mean = np.mean(val_accuracy_best_folds)
df_results_all.to_csv(f'results cyclegan {args.model} {args.lr}/{args.data_epoch}/{args.data_patched_generated}/results_{figure_name}_({val_acc_best_mean:.1f}).csv', index=False)
save_train_test_plot_folds(train_loss_folds, val_loss_folds, 'Loss', figure_name_loss)
save_train_test_plot_folds(train_accuracy_folds, val_accuracy_folds, 'Acc', figure_name_acc)
