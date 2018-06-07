from __future__ import print_function
import argparse
import os
import torch
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from munch import Munch
from models import *
from utils.feature_inversion_utils import *
from utils.perceptual_loss.perceptual_loss import get_pretrained_net
from utils.common_utils import *
 
# Setup Cuda
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configuration
conf = Munch()
conf.pretrained_net = 'alexnet_caffe'
conf.layer_to_invert = 'fc6'
conf.data_type = torch.cuda.FloatTensor

# Get the pre-trained model
cnn = get_pretrained_net(conf.pretrained_net)
cnn = cnn.type(conf.data_type)

# Remove the layers we don't need 
target_layer_index = list(cnn._modules.keys()).index('fc6')
cnn = nn.Sequential(*list(cnn)[:target_layer_index+1])   
print(cnn)