from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import numpy as np

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





