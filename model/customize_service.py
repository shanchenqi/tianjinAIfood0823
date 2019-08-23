
#!/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os
from math import exp
import numpy as np

from PIL import Image

from model_service.pytorch_model_service import PTServingBaseService
import torch.nn as nn
import torch
import logging

import torchvision.transforms as transforms


infer_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
MODEL_OUTPUT_KEY = 'logits'
LABELS_FILE_NAME = 'small_labels_25c.txt'
ALL_LIST_NAME = 'all_label.txt'

def decode_image(file_content):

    """

    Decode bytes to a single image

    :param file_content: bytes

    :return: ndarray with rank=3

    """

    image = Image.open(file_content)
    image = image.convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    return image

def read_label_list(path):

    """

    read label list from path
    :param path: a path
    :return: a list of label names like: ['label_a', 'label_b', ...]
    """
    with open(path, 'r') as f:
        label_list = f.read().split(os.linesep)
    label_list = [x.strip() for x in label_list if x.strip()]
    return label_list


class FoodPredictService(PTServingBaseService):
    def __init__(self, model_name, model_path):

        global LABEL_LIST
        global ALL_LIST
        super(FoodPredictService, self).__init__(model_name, model_path)
        self.model = resnet50(model_path)
        dir_path = os.path.dirname(os.path.realpath(self.model_path))

        LABEL_LIST = read_label_list(os.path.join(dir_path, LABELS_FILE_NAME))
        ALL_LIST = read_label_list(os.path.join(dir_path, ALL_LIST_NAME))

    def _preprocess(self, data):

        """
        `data` is provided by Upredict service according to the input data. Which is like:
          {

              'images': {

                'image_a.jpg': b'xxx'

              }

          }

        For now, predict a single image at a time.

        """

        preprocessed_data = {}
        input_batch = []

        for file_name, file_content in data[IMAGES_KEY].items():
            

            print('\tAppending image: %s' % file_name)

            image1 = decode_image(file_content)



            if torch.cuda.is_available():

                input_batch.append(infer_transformation(image1).cuda())

            else:

                input_batch.append(infer_transformation(image1))


        input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)

        preprocessed_data[MODEL_INPUT_KEY] = input_batch_var


        return preprocessed_data



    def _postprocess(self, data):

        print('data:', data)

        """
        `data` is the result of your model. Which is like:
          {
            'logits': [[0.1, -0.12, 0.72, ...]]
          }
        value of logits is a single list of list because one image is predicted at a time for now.

        """

        # logits_list = [0.1, -0.12, 0.72, ...]
    
        logits_list = data['images'][0].detach().numpy().tolist()
   
        print('LABEL_LIST', LABEL_LIST)
        print('logits_list', logits_list)

        labels_to_logits = {

            ALL_LIST[i]: s for i, s in enumerate(logits_list) if ALL_LIST[i] in LABEL_LIST

        }

        lis = sorted([(label, possible) for label, possible in labels_to_logits.items()], key = lambda x: -x[1])

        labels_to_logits = {label:possible for label, possible in lis}
      
        predict_result = {

            MODEL_OUTPUT_KEY: labels_to_logits
           

        }
        print (predict_result)

        return predict_result


def conv3x3(in_planes, out_planes, stride=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=1, bias=False)



def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    expansion = 1


    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)



        out = self.conv2(out)
        out = self.bn2(out)



        if self.downsample is not None:
            identity = self.downsample(x)



        out += identity
        out = self.relu(out)



        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)

        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out






class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=75, zero_init_residual=False):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes))



        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def resnet50(model_path, **kwargs):

    """Constructs a ResNet-50 model.


    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    model.load_state_dict(torch.load(model_path,map_location ='cpu'))

    model.eval()

    return model