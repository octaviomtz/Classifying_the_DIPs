import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from torch import optim
from itertools import groupby, count
from scipy.signal import savgol_filter
from torch import optim
from torch.autograd import Variable

from lr_finder import *

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

def save_train_test_plot_folds(train_folds_, test_folds_, acc_or_loss, title):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    for idx, (train_, test_) in enumerate(zip(train_folds_, test_folds_)):
        if idx==0:
            ax.plot(train_, c=colors[0], label="train")
            ax.plot(test_, c=colors[1], label='test')
        else:
            ax.plot(train_, c=colors[0])
            ax.plot(test_, c=colors[1])
    ax.plot(test_, c=colors[1])
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
        img_ = copy.copy(img_)
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

def class_imbalance_sampler(labels):
    '''get class imbalance from labels and return a sampler pytorch object'''
    train_imbalance = torch.from_numpy(np.asarray(labels)*1)
    # Get the weight of each class and assign the corresponding weight to each element to the
    weight_class1 = len(train_imbalance)/sum(train_imbalance).item() 
    weight_class0 = len(train_imbalance)/(len(train_imbalance)-sum(train_imbalance)).item()
    weights_array = []
    for i in train_imbalance:
        if i==1: weights_array.append(weight_class1)
        else: weights_array.append(weight_class0)        
    sampler = WeightedRandomSampler(weights_array, len(weights_array))
    return sampler

def weights_init(m):
    '''Xavier initialization of Conv2d layers'''
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# LEARNING RATE FINDER
def find_lr_get_losses(model_, optimizer_, criterion_, dataloader_train_):
    '''LR finder: Evaluate different learning rates (incrementally larger) and return their losses'''
    optimizer = optim.Adam(model_.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model_, optimizer_, criterion_, device="cuda")
    lr_finder.range_test(dataloader_train_, end_lr=.1, num_iter=100)
    lr_finder_lr = np.asarray(lr_finder.history['lr'])
    lr_finder_loss = np.asarray(lr_finder.history['loss'])
    lr_finder.reset()
    return lr_finder_lr, lr_finder_loss

def find_lr_get_lr(lr_finder_lr_, lr_finder_loss_):
    '''Get (almost) the largest LR from the longest consecutive negative slope of the (filtered) losses.
    Returns:
    learning rate, learning rate index (from the lr evaluated), filtered lr losses, 
    indices of longest consecutive negative slope of the (filtered) losses (idx_neg_slp)'''
    # Apply Savitzky-Golay filter https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    lr_finder_loss_filtered_ = savgol_filter(lr_finder_loss_, 25, 2)
    
    loss_going_down = np.where(np.diff(lr_finder_loss_filtered_) < -1e-7) # indices that go down (negative diff)
    loss_going_down = list(loss_going_down[0] + 1)
    c = count()
    idx_neg_slp = max((list(g) for _, g in groupby(loss_going_down, lambda x: x-next(c))), key=len) # longest sequence of negative diff
    idx_neg_slp = list(idx_neg_slp)
    slope_diff = np.diff(lr_finder_loss_filtered_[idx_neg_slp])
    largest_diff = np.where(slope_diff == np.min(slope_diff))[0]
    LR_idx = idx_neg_slp[largest_diff[0]]
    LR = lr_finder_lr_[LR_idx]
    return LR, LR_idx, lr_finder_loss_filtered_, idx_neg_slp

def loss_batch(model, loss_func, xb, yb, device, opt=None):
    '''Based on https://pytorch.org/tutorials/beginner/nn_tutorial.html'''
    if torch.cuda.is_available():
                xb = Variable(xb.cuda(device))
                yb = Variable(yb.cuda(device))
    else:
        xb = Variable(xb)
        yb = Variable(yb)

    pred = model(xb)
    loss = loss_func(pred, yb)
    _, pred_class = torch.max(pred.data, 1)
    batch_total = yb.size(0)

    # Accuracies
    if torch.cuda.is_available():
        batch_correct = (pred_class.cpu() == yb.cpu()).sum().item()
    else:
        batch_correct = (pred_class == yb).sum().item()

    return pred, loss, batch_total, batch_correct, pred_class