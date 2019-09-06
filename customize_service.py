
#!/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os
from math import exp
import numpy as np

from PIL import Image
import cv2
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn as nn
import torch
import logging
import torchvision.models as models
import torchvision.transforms as transforms


infer_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
MODEL_OUTPUT_KEY = 'logits'
LABELS_FILE_NAME = 'small_labels_25c.txt'


def decode_image(file_content):

    """

    Decode bytes to a single image

    :param file_content: bytes

    :return: ndarray with rank=3

    """

    image = Image.open(file_content)
    image = image.convert('RGB')
    # print(image.shape)
   # image = np.asarray(image, dtype=np.float32)
    return image
#    image_content = r.files[file_content].read() # python 'List' class that holds byte
#    np_array = np.fromstring(image_content, np.uint8) # numpy array with dtype np.unit8
#    img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # numpy array in shape [height, width, channels]
 

def read_label_list(path):

    """

    read label list from path
    :param path: a path
    :return: a list of label names like: ['label_a', 'label_b', ...]
    """
    with open(path, 'r') as f:
        label_list = f.read().split(os.linesep)
    label_list = [x.strip() for x in label_list if x.strip()]
    print(' label_list',label_list)
    return label_list


class FoodPredictService(PTServingBaseService):
    def __init__(self, model_name, model_path):

        global LABEL_LIST
        super(FoodPredictService, self).__init__(model_name, model_path)
        self.model = resnet50(model_path)
        dir_path = os.path.dirname(os.path.realpath(self.model_path))

        LABEL_LIST = read_label_list(os.path.join(dir_path, LABELS_FILE_NAME))

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

       # print('preprocessed_data',input_batch_var.shape())

        return preprocessed_data

    def _postprocess(self, data):

        """

        `data` is the result of your model. Which is like:

          {

            'logits': [[0.1, -0.12, 0.72, ...]]

          }

        value of logits is a single list of list because one image is predicted at a time for now.

        """

        # logits_list = [0.1, -0.12, 0.72, ...]

        logits_list =  data['images'][0].detach().numpy().tolist()
        maxlist=max(logits_list)
        z_exp = [exp(i-maxlist) for i in  logits_list]
        
        sum_z_exp = sum(z_exp)
        softmax = [round(i / sum_z_exp, 3) for i in z_exp]


        # labels_to_logits = {

        #     'label_a': 0.1, 'label_b': -0.12, 'label_c': 0.72, ...

        # }

        labels_to_logits = {
            LABEL_LIST[i]: s for i, s in enumerate(softmax)
            # LABEL_LIST[i]: s for i, s in enumerate(logits_list)

        }


        predict_result = {

            MODEL_OUTPUT_KEY: labels_to_logits

        }

        return predict_result


def resnet50(model_path, **kwargs):

    """Constructs a ResNet-50 model.


    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 25)
    model.load_state_dict(torch.load(model_path,map_location ='cpu'))
    # model.load_state_dict(torch.load(model_path))

    model.eval()

    return model