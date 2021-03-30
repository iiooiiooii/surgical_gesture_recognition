import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.init import normal, constant

from resnet3D import resnet18
from transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from train_opts import num_cls_Kinetics


