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
            std = 0.001
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
            
            if self.modality == 'Flow':
                self.base_model = self._construct_flow_model(self.base_model)

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    def _add_upsampling_layers(self, num_class):
        in_channels = None
        if self.arch == "3D-ResNet-18":
            in_channels = 512


        self.up_conv = nn.ConvTranspose1d(in_channels, num_class, 11, stride=5, padding=0, output_padding=0,
                                          groups=1, bias=True, dilation=1)
        
        for module in self.up_conv.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.001)

                    
    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.snippet_length
        
        if not self.is_3D_architecture:
            input = input.view((-1, sample_len) + input.size()[-2:])

        base_out = self.base_model(input)
        if not self.is_3D_architecture:
            if self.new_fc is not None:
                base_out = self.new_fc(base_out)
            out = base_out
        else:
            out = base_out.view(base_out.size(0), base_out.size(1), -1)
            out = self.up_conv(out)

        return out
    def crop_size(self):
        return self.input_size
    
    def scale_size(self):
        return self.input_size * 256 // 224
    
    def is_3D_architecture(self):
        return '3d' in self.arch.casefold()

    def get_augmentation(self, crop_corners=True, do_horizontal_flip=True):
        if do_horizontal_flip:
            if self.modality == 'RGB':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                           fix_crop=crop_corners,
                                                                           more_fix_crop=crop_corners),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            elif self.modality == 'Flow':
            elif self.modality == 'Flow':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75],
                                                                           fix_crop=crop_corners,
                                                                           more_fix_crop=crop_corners),
                                                       GroupRandomHorizontalFlip(is_flow=True)])
        else:
            if self.modality == 'RGB':
               return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                           fix_crop=crop_corners,
                                                                           more_fix_crop=crop_corners)])
            elif self.modality == 'Flow':
               return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75],
                                                                           fix_crop=crop_corners,
                                                                           more_fix_crop=crop_corners)])
   """
    https://github.com/iiooiiooii/surgical_gesture_recognition/edit/main/models.py
   """
    def _construct_flow_model(self, base_model, in_channels=-1):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
