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
from utils.feature_inversion_utils import *
from utils.perceptual_loss.perceptual_loss import get_pretrained_net
from utils.common_utils import *
from utils.feature_inversion_utils import View

# Configuration
conf = Munch()
conf.pretrained_net = 'alexnet_caffe'
#conf.pretrained_net = 'alexnet_torch'
conf.layer_to_invert = 'fc6'
conf.data_type = torch.FloatTensor
conf.pad = 'zero'
conf.optimizer = 'adam'
conf.lr = 0.001
conf.num_iter = 3100
conf.input_type = 'noise'
conf.input_depth = 32
conf.plot = False
conf.cuda = '0'

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
    last = layers.index(conf.layer_to_invert)
    for k in layers[last+1:]:
        cnn._modules.pop(k)
    print(cnn)

def invert(conf, cnn, ims): 

    # Make sure it is alist
    singleton = type(ims) != list
    if singleton:
        ims = [ims]

    # Load image to process
    imsize_net = 256
    imsize = 227 if conf.pretrained_net == 'alexnet' else 224
    preprocess, deprocess = get_preprocessor(imsize), get_deprocessor()
    ims0 = []
    ims0_preprocessed = []
    for im in ims:
        w, h = im.size
        dx, dy = (w - imsize)/2, (h - imsize)/2
        im = im.crop(box=(dx, dy, w - dx, h - dy))
        im_preprocessed = preprocess(im)
        ims0_preprocessed.append(im_preprocessed[None,:]) # 3 x H x W -> 1 x 3 x H x W
        ims0.append(deprocess(im_preprocessed))

    # Matcher: store target feature values for inversions
    matcher_opts = {'layers': conf.layer_to_invert, 'what': 'features'}
    matcher = get_matcher(cnn, matcher_opts)
    matcher.mode = 'store'
    cnn(torch.cat(ims0_preprocessed, 0).type(conf.data_type))

    # Generator network (prior)
    net_input = get_noise(conf.input_depth, conf.input_type, imsize_net, card = len(ims))
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
    invert.iteration = 0
    def train_callback():        
        generated = net(net_input)[:, :, :imsize, :imsize]
        generated_preprocessed = vgg_preprocess_var(generated)
        cnn(generated_preprocessed)
        total_loss = sum(matcher.losses.values())
        total_loss.backward()
        print ('Iteration %05d    Loss %.3f' %
            (invert.iteration, total_loss.item()), '\r', end='')
        if conf.plot and invert.iteration % 200 == 0:
            generated_np = [np.clip(torch_to_np(x), 0, 1) for x in torch.chunk(generated, len(ims))]
            plot_image_grid(generated_np, 8, 1, num=1)
            plt.pause(0.001)
        invert.iteration += 1
        return total_loss

    matcher.mode = 'match'
    p = get_params('net', net, net_input)
    optimize(conf.optimizer, p, train_callback, conf.lr, conf.num_iter)

    generated = net(net_input)[:, :, :imsize, :imsize]
    generated_np = [np.clip(torch_to_np(x), 0, 1) for x in torch.chunk(generated, len(ims))]
    ims1 = [Image.fromarray((255 * x).astype(np.uint8).transpose(1, 2, 0)) for x in generated_np]
    if singleton:
        ims0, ims1 = ims0[0], ims1[0]
    return ims1, ims0