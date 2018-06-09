from __future__ import print_function
import argparse
import os
import glob
import torch
import torch.optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from munch import Munch
from models import *
from utils.perceptual_loss.matcher import *
from utils.perceptual_loss.perceptual_loss import get_pretrained_net, get_matcher, vgg_preprocess_caffe
from utils.common_utils import *
from utils.feature_inversion_utils import View

# Configuration
conf = Munch()
conf.pretrained_net = 'alexnet_caffe'
#conf.pretrained_net = 'alexnet_torch'
conf.layer_to_maximize = 'fc8'
conf.data_type = torch.FloatTensor
conf.pad = 'zero'
conf.optimizer = 'adam'
conf.lr = 0.01
conf.weight_decay = 1
conf.num_iter = 2500
conf.input_type = 'noise'
conf.input_depth = 32
conf.plot = True
conf.cuda = '1'

def xmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_net(conf):
    # Setup Cuda
    if conf.cuda is not None:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUDA_VISIBLE_DEVICES'] = conf.cuda
        conf.data_type = torch.cuda.FloatTensor

    # Get the pre-trained model, removing layers we do not need
    cnn = get_pretrained_net(conf.pretrained_net).type(conf.data_type)
    return cnn

def slice_net(conf, cnn):
    # Get the pre-trained model, removing layers we do not need
    cnn = get_pretrained_net(conf.pretrained_net).type(conf.data_type)
    layers = list(cnn._modules.keys())
    last = layers.index(conf.layer_to_maximize)
    for k in layers[last+1:]:
        cnn._modules.pop(k)
    print(cnn)

def get_neuron_for_class(class_name):
    """Return a class that contains `class_name` in its name."""
    import json
    with open('data/imagenet1000_clsid_to_human.txt', 'r') as f:
        corresp = json.load(f)
    for index, name in corresp.items():
        if class_name in name:
            return int(index)
    return None

def maximize(conf, cnn, neuron): 

    # Get image size
    imsize = 227 if conf.pretrained_net == 'alexnet' else 224
    imsize_net = 256

    # Matcher: store target feature values for inversions
    matcher_opts = {
        'layers': [conf.layer_to_maximize],
        'what': 'features', 
        'map_idx': neuron}
    matcher = get_matcher(cnn, matcher_opts)
    matcher.mode = 'match'

    # Generator network (prior)
    net_input = get_noise(conf.input_depth, conf.input_type, imsize_net, card = 1)
    net_input = net_input.type(conf.data_type).detach()
    net = skip(conf.input_depth, 3,
        num_channels_down = [16, 32, 64, 128, 128, 128],
        num_channels_up   = [16, 32, 64, 128, 128, 128],
        num_channels_skip = [ 4,  4,  4,   4,   4,   4],
        filter_size_down  = [ 7,  7,  5,   5,   3,   3],
        filter_size_up    = [ 7,  7,  5,   5,   3,   3],
        upsample_mode = 'nearest',
        downsample_mode = 'avg',
        need_sigmoid = True,
        pad = conf.pad,
        act_fun = 'LeakyReLU')
    net = net.type(conf.data_type)

    # Optimisation
    maximize.iteration = 0
    def train_callback():       
        generated = net(net_input)[:, :, :imsize, :imsize] 
        generated_preprocessed = vgg_preprocess_caffe(generated)
        cnn(generated_preprocessed)
        total_loss = sum(matcher.losses.values())
        total_loss.backward()
        print ('Iteration %05d    Loss %.3f' %
            (maximize.iteration, total_loss.item()), '\r', end='')
        if conf.plot and maximize.iteration % 200 == 0:
            generated_np = [np.clip(torch_to_np(x), 0, 1) for x in torch.chunk(generated, 1)]
            plot_image_grid(generated_np, 8, 1, num=1)
            plt.pause(0.001)
        maximize.iteration += 1
        return total_loss

    matcher.method = 'maximize'
    matcher.method = 'match'
    p = get_params('net', net, net_input)
    optimize(conf.optimizer, p, train_callback, LR=conf.lr, num_iter=conf.num_iter, weight_decay=conf.weight_decay)
    
    generated = net(net_input)[:, :, :imsize, :imsize]
    generated_np = np.clip(torch_to_np(generated), 0, 1)
    im = Image.fromarray((255 * generated_np).astype(np.uint8).transpose(1, 2, 0))
    return im
   