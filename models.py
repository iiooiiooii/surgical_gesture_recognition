import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.init import normal, constant

from resnet3D import resnet18
from transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from train_opts import num_cls_Kinetics


class GestureClassifier(nn.Module):
    def __init__(self, base_model, num_class, modality, dropout=0.8, snippet_length=16, input_size=112,
                 pretrained_model=None, bootstrap_from_2D=False, use_resnet_shortcut_type_B=False):
        super(GestureClassifier, self).__init__()
        self.arch = base_model
        self.modality = modality
        
        self._init_base_model(base_model, num_class, dropout, snippet_length, input_size,
                              pretrained_model, bootstrap_from_2D, use_resnet_shortcut_type_B)

    def _init_base_model(self, base_model, num_class, dropout=0.8, snippet_length=16, input_size=112,
                         pretrained_model=None, bootstrap_from_2D=False, use_resnet_shortcut_type_B=False):
        if base_model == '3D-ResNet-18':
            self.snippet_length = snippet_length
            self.input_size = input_size
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]

