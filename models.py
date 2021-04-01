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
                

            self.base_model = resnet18(
                sample_size=self.input_size,
                sample_duration=self.snippet_length,
                shortcut_type=('B' if use_resnet_shortcut_type_B else 'A'),
                num_classes=num_cls_Kinetics,
                skip_classifier=True,
                no_initial_temporal_pooling=True)
            
            if pretrained_model:
                print('loading pretrained model {}'.format(pretrained_model))

               if bootstrap_from_2D:
                    model_weights = torch.load(pretrained_model)
                    model_weights = {k if 'base_model' not in k else '.'.join(k.split('.')[1:]): v
                                     for k, v in list(model_weights.items())}
                    print("inflate 2D weights...")
                    for k in model_weights:
                        tensor = model_weights[k]
                        if len(tensor.shape) == 4:
                            model_weights[k] = inflate_weights(tensor, tensor.shape[-1])  # symmetric kernel
                    self.base_model.load_state_dict(model_weights, strict=False)
                else:
                    checkpoint = torch.load(pretrained_model)
                    assert checkpoint['arch'] == "resnet-18"
                    model_weights = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
                    self.base_model.load_state_dict(model_weights, strict=False)

            self._add_upsampling_layers(num_class)

        elif base_model == 'resnet18':  # 2D baseline model
            if self.modality == 'RGB':
                assert(snippet_length == 1)
            elif self.modality == 'Flow':
                assert(snippet_length == 5)
            self.snippet_length = snippet_length
            self.input_size = input_size
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]

            self.base_model = torchvision.models.resnet18(pretrained=True)
            
            # adapt base model
            last_layer_name = 'fc'
            feature_dim = getattr(self.base_model, last_layer_name).in_features
            setattr(self.base_model, last_layer_name, nn.Dropout(p=dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)
