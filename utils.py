from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import numpy as np
import os

def CustomCmap(low,top):
    r1,g1,b1 = low
    r2,g2,b2 = top
    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}
    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def plot_confusion_matrix(y_true, y_pred, cmap, target_names, title = None,
            ignore_index = False):
    fig, ax1 = plt.subplots(figsize = (8,8))
    cfmatrix = confusion_matrix(y_true = y_true, y_pred = y_pred) if ignore_index else\
        confusion_matrix(y_true = y_true-1, y_pred = y_pred-1)

    for ax,cm in zip([ax1],[cfmatrix]):
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=.2)
        plt.colorbar(im, cax=cax) #, ticks=[-1,-0.5,0,0.5,1]
        ax.set_title(title,fontsize=14)
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

        fmt = 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label',fontsize=14)
        ax.set_xlabel('Predicted label',fontsize=14)
    plt.show()
    
    
import torch.nn as nn
import torch
from typing import OrderedDict
from torch.utils.data import Dataset

class MLP(nn.Module):
    def __init__(self, in_features = 28*28, out_features = 10,
                hidden_layers = [512,256,128]):
        super(MLP,self).__init__()
        self.structure = OrderedDict()
        for i,layer in enumerate(hidden_layers):
            if i == 0:
                self.structure["Linear " + str(i+1)] = nn.Linear(in_features=in_features,
                                                                    out_features=layer, bias=False)
                self.structure["BatchNorm1D " + str(i+1)] = nn.BatchNorm1d(layer)
                self.structure["Relu " + str(i+1)] = nn.ReLU()
                self.structure["Dropout " + str(i+1)] = nn.Dropout(p = 0.2)
            else:
                self.structure["Linear " + str(i+1)] = nn.Linear(in_features = hidden_layers[i-1],
                                                    out_features=layer, bias=False)
                self.structure["BatchNorm1D " + str(i+1)] = nn.BatchNorm1d(layer)
                self.structure["Relu " + str(i+1)] = nn.ReLU()
                self.structure["Dropout " + str(i+1)] = nn.Dropout(p = 0.2)
        self.structure["Out Linear"] = nn.Linear(in_features=hidden_layers[-1],out_features=out_features)
        
        self.linear_relu_stack = nn.Sequential(self.structure)
        
    def forward(self,x):
        return self.linear_relu_stack(x)

class mlp_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx,:], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.int64)




def custom_split(features, labels, test_ratio = 0.15, print_progress = False,
                 random_state = 42):
    num_classes = np.unique(labels)
    X_train, y_train = np.empty(shape = (0,len(features[0,:])), dtype = np.float64), np.array([], dtype=np.int64)
    X_test, y_test = np.empty(shape = (0,len(features[0,:])), dtype = np.float64), np.array([], dtype=np.int64)
    for class_label in num_classes:
        indices = np.where(labels == class_label)[0]
        num_test_samples = int(test_ratio*len(indices))
        np.random.seed(random_state)
        test_indices = np.random.choice(indices, replace = False,
                                        size = num_test_samples)
        train_indices = np.setdiff1d(indices, test_indices,
                                     assume_unique = True)
        X_train = np.concatenate((X_train, features[train_indices,:]), axis = 0)
        X_test = np.concatenate((X_test, features[test_indices,:]), axis = 0)
        y_train = np.concatenate((labels[train_indices], y_train))
        y_test = np.concatenate((labels[test_indices], y_test))
        if print_progress:
            print(f"- Class {class_label}: Train|Test ---> {len(train_indices)}|{len(test_indices)}")
    return X_train, X_test, y_train, y_test
        
        

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 5, stride = 1, padding = 0, bias = True):
        super(conv_block,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels =  out_channels,
                      kernel_size = kernel_size, stride = stride,
                      padding = padding, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        
    def forward(self,x): 
        return self.conv(x)
class CNN(nn.Module):
    def __init__(self,in_channels = 3, out_channels= 10):
        super(CNN,self).__init__()
        self.backbone = nn.Sequential(
            conv_block(in_channels=in_channels,
                       out_channels=6, bias = False),
            nn.AdaptiveMaxPool2d(output_size = 62),
            conv_block(in_channels = 6, out_channels = 8,
                       bias = False),
            nn.AdaptiveMaxPool2d(output_size = 29),
            conv_block(in_channels=8, out_channels=12,
                       bias=False),
            nn.AdaptiveAvgPool2d(output_size= 12),
            conv_block(in_channels=12, out_channels=16,
                       bias=False),
            nn.AdaptiveAvgPool2d(output_size = 4),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*16, out_features=128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 64, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 10),
        )
    
    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)

import glob
import cv2

class cnn_dataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X,y
        self.features = []
        for img_path in self.X:
            img = cv2.imread(img_path)
            img = cv2.resize(img, dsize = (128,128))
            self.features.append(np.transpose(img, (2,0,1)))
        
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        img = self.features[idx] # Transpose from HxWxC --> CxHxW
        return torch.tensor(img, dtype = torch.float32)/255, torch.tensor(self.y[idx], dtype = torch.int64)

def split_images(imgs_path, ratios = [0.70, 0.15, 0.15]):
    genres = list(os.walk(imgs_path))[0][1]
    class_mapping = {"blues":0, "classical":1, "country":2, "disco":3,
                     "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock":9}
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    
    for genre in genres:
        imgs = np.array(list(glob.glob(os.path.join(imgs_path,
                                           genre, "*.png"))))
        idxs = np.arange(start =0, stop = len(imgs))
        train_idxs = np.random.choice(a = idxs, replace = False,
                                      size = int(ratios[0]*len(idxs)))
        rest_idxs = np.setdiff1d(ar1 = idxs,
                                 ar2 = train_idxs, assume_unique=True)
        alpha = ratios[1]/ratios[2]
        val_ratio = 1/(1+alpha)
        val_idxs = np.random.choice(a = rest_idxs, replace =  False,
                                    size = int(val_ratio*len(rest_idxs)))
        test_idxs = np.setdiff1d(ar1 = rest_idxs, ar2 = val_idxs,
                                 assume_unique=True)
        y_train += [class_mapping[genre]]*len(train_idxs)
        y_val += [class_mapping[genre]]*len(val_idxs)
        y_test += [class_mapping[genre]]*len(test_idxs)
        X_train += list(imgs[train_idxs])
        X_val += list(imgs[val_idxs])
        X_test += list(imgs[test_idxs])
        
    target_names = 10*[0]
    for k,v in class_mapping.items():
        target_names[v] = k
    
    return X_train, y_train, X_val, y_val, X_test, y_val, target_names
        
