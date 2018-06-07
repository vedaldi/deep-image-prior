# coding: utf-8

# In[1]:

from __future__ import print_function
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from models import *

import torch
import torch.optim

from utils.feature_inversion_utils import *
from utils.perceptual_loss.perceptual_loss import get_pretrained_net
from utils.common_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
fname = './data/feature_inversion/building.jpg'

pretrained_net = 'alexnet_caffe' # 'vgg19_caffe'
layers_to_use = 'fc6' # comma-separated string of layer names e.g. 'fc6,fc7'

# In[2]:
# Run pretrained model

cnn = get_pretrained_net(pretrained_net).type(dtype)

opt_content = {'layers': layers_to_use, 'what':'features'}

# Remove the layers we don't need 
keys = [x for x in cnn._modules.keys()]
max_idx = max(keys.index(x) for x in opt_content['layers'].split(','))
for k in keys[max_idx+1:]:
    cnn._modules.pop(k)
    
print(cnn)

# In[3]:

# Target imsize 
imsize = 227 if pretrained_net == 'alexnet' else 224

# Something divisible by a power of two
imsize_net = 256

# VGG and Alexnet need input to be correctly normalized
preprocess, deprocess = get_preprocessor(imsize), get_deprocessor()

img_content_pil, img_content_np  = get_image(fname, imsize)
img_content_prerocessed = preprocess(img_content_pil)[None,:].type(dtype)
img_content_pil

# # Setup matcher and net

# In[4]:
matcher_content = get_matcher(cnn, opt_content)
matcher_content.mode = 'store'
cnn(img_content_prerocessed)

# In[5]:
INPUT = 'noise'
pad = 'zero' # 'refection'
OPT_OVER = 'net' #'net,input'
OPTIMIZER = 'adam' # 'LBFGS'
LR = 0.001
num_iter = 3100
input_depth = 32
net_input = get_noise(input_depth, INPUT, imsize_net).type(dtype).detach()
# In[6]:
net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [4, 4, 4, 4, 4, 4],   
                           filter_size_down = [7, 7, 5, 5, 3, 3], filter_size_up = [7, 7, 5, 5, 3, 3], 
                           upsample_mode='nearest', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# # Optimize
# In[7]:
def closure():
    
    global i
           
    out = net(net_input)[:, :, :imsize, :imsize]
    
    cnn(vgg_preprocess_var(out))
    total_loss =  sum(matcher_content.losses.values())
    total_loss.backward()
    
    print ('Iteration %05d    Loss %.3f' % (i, total_loss.item()), '\r', end='')
    if PLOT and i % 200 == 0:
        out_np = np.clip(torch_to_np(out), 0, 1)
        plot_image_grid([out_np], 3, 3, num=1);
        plt.pause(0.001)

    i += 1
    
    return total_loss

# In[8]:
i=0
matcher_content.mode = 'match'
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

# # Result
# In[9]:
out = net(net_input)[:, :, :imsize, :imsize]
plot_image_grid([torch_to_np(out)], 3, 3);

# The code above was used to produce the images from the paper.

# # Appedndix: more noise

# We also found adding heavy noise sometimes improves the results (see below). Interestingly, network manages to adapt to a very heavy noise.

# In[10]:
input_depth = 2
net_input = get_noise(input_depth, INPUT, imsize_net).type(dtype).detach()
net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [4, 4, 4, 4, 4, 4],   
                           filter_size_up = [7, 7, 5, 5, 3, 3], filter_size_down = [7, 7, 5, 5, 3, 3],
                           upsample_mode='nearest', downsample_mode='avg',
                           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# In[11]:
def closure():
    global i    
    if i < 10000:
        # Weight noise
        for n in [x for x in net.parameters() if len(x) == 4]:
            n = n + n.detach().clone().normal_()*n.std()/50
        
        # Input noise
        net_input = net_input_saved + (noise.normal_() * 10)

    elif i < 15000:
        # Weight noise
        for n in [x for x in net.parameters() if len(x) == 4]:
            n = n + n.detach().clone().normal_()*n.std()/100
        
        # Input noise
        net_input = net_input_saved + (noise.normal_() * 2)
        
    elif i < 20000:
        # Input noise
        net_input = net_input_saved + (noise.normal_() / 2)
        
    out = net(net_input)[:, :, :imsize, :imsize]
    
    cnn(vgg_preprocess_var(out))
    total_loss =  sum(matcher_content.losses.values())
    total_loss.backward()
    
    print ('Iteration %05d    Loss %.3f' % (i, total_loss.item()), '\r', end='')
    if PLOT and i % 1000 == 0:
        out_np = np.clip(torch_to_np(out), 0, 1)
        plot_image_grid([out_np], 3, 3, num=1);
        np.pause(0.001)
    
    i += 1
    
    return total_loss


# In[ ]:
num_iter = 20000
LR = 0.01

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
i=0

matcher_content.mode = 'match'
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

