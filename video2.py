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
conf.pad = 'zero'
conf.optmizer = 'adam'
conf.lr = 0.001
conf.num_iter = 3100
conf.input_depth = 32
conf.plot = True

# Get the pre-trained model
cnn = get_pretrained_net(conf.pretrained_net)
cnn = cnn.type(conf.data_type)

# Remove the layers we don't need 
target_layer_index = list(cnn._modules.keys()).index('fc6')
cnn = nn.Sequential(*list(cnn)[:target_layer_index+1])   
print(cnn)

# Load image preprocessor
imsize = 227 if pretrained_net == 'alexnet' else 224
imsize_net = 256
preprocess, deprocess = get_preprocessor(imsize), get_deprocessor()

# Load and normalise image
img_content_pil, img_content_np  = get_image(fname, imsize)
img_content_prerocessed = preprocess(img_content_pil)[None,:]
img_content_prerocessed = img_content_preprocessed.type(conf.data_type)
img_content_pil

# Matcher net
matcher_content = get_matcher(cnn, opt_content)
matcher_content.mode = 'store'
cnn(img_content_prerocessed)

# Input ?
net_input = get_noise(input_depth, INPUT, imsize_net).type(dtype).detach()

# Generator network (prior)
net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [4, 4, 4, 4, 4, 4],   
                           filter_size_down = [7, 7, 5, 5, 3, 3],
                           filter_size_up = [7, 7, 5, 5, 3, 3], 
                           upsample_mode = 'nearest',
                           downsample_mode = 'avg',
                           need_sigmoid = True, pad = pad, act_fun = 'LeakyReLU')
net = net.type(conf.data_type)

# Optimisation
def train_callback():    
    global iteration 
    out = net(net_input)[:, :, :imsize, :imsize]    
    cnn(vgg_preprocess_var(out))
    total_loss =  sum(matcher_content.losses.values())
    total_loss.backward()    
    print ('Iteration %05d    Loss %.3f' %
        (iteration, total_loss.item()), '\r', end='')
    if conf.plot and i % 200 == 0:
        out_np = np.clip(torch_to_np(out), 0, 1)
        plot_image_grid([out_np], 3, 3, num=1)
        plt.pause(0.001)
    iteration += 1    
    return total_loss

iteration = 0
matcher_content.mode = 'match'
p = get_params('net', net, net_input)
optimize(conf.optimiser, p, closure, LR, num_iter)

# Final result
out = net(net_input)[:, :, :imsize, :imsize]
plot_image_grid([torch_to_np(out)], 3, 3)