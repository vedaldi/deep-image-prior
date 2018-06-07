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

fname = './data/feature_inversion/building.jpg'

# Configuration
conf = Munch()
conf.pretrained_net = 'alexnet_caffe'
conf.layer_to_invert = 'fc6'
conf.data_type = torch.cuda.FloatTensor
#conf.data_type = torch.FloatTensor
conf.pad = 'zero'
conf.optimizer = 'adam'
conf.lr = 0.001
conf.num_iter = 3100
conf.input_type = 'noise'
conf.input_depth = 32
conf.plot = False
conf.cuda = '0'

# What to do based on sysarg
if len(sys.argv) > 1:
    case = int(sys.argv[1])
    if case == 1:
        conf.layer_to_invert = 'fc6'
        conf.cuda = '0'
    elif case == 2:
        conf.layer_to_invert = 'conv5'
        conf.cuda = '2'
    elif case == 3:
        conf.layer_to_invert = 'fc8'
        conf.cuda = '3'

# Setup Cuda
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES'] = conf.cuda

prefix = 'data/blue'

# Load image preprocessor
imsize = 227 if conf.pretrained_net == 'alexnet' else 224
imsize_net = 256
preprocess, deprocess = get_preprocessor(imsize), get_deprocessor()
def get_normalized_image(path):
    im = load(path)
    w, h = im.size
    dx, dy = (w - imsize)/2, (h - imsize)/2
    im = im.crop(box=(dx, dy, w - dx, h - dy))
    im = deprocess(preprocess(im))
    return im

# Create destination folder
output_folder = os.path.join(prefix, conf.layer_to_invert)
input_folder = os.path.join(prefix, 'x0')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(input_folder):
    os.mkdir(input_folder)

# Get the pre-trained model, removing layers we do not need
cnn = get_pretrained_net(conf.pretrained_net).type(conf.data_type)
layers = list(cnn._modules.keys())
last = layers.index(conf.layer_to_invert)
for k in layers[last+1:]:
    cnn._modules.pop(k)
print(cnn)

# Load and normalise image
for file in glob.glob(os.path.join(prefix, "*.jpg")):
    input_file = os.path.join(input_folder, os.path.basename(file))
    output_file = os.path.join(output_folder, os.path.basename(file))
    print(file, output_file)        
    im_reference = get_normalized_image(file)
    im_reference.save(input_file)
    if os.path.exists(output_file):
        print("Skipping because it already exists.")
        continue
    
    # Matcher: store target feature values for inversions
    opt_content = {'layers': conf.layer_to_invert, 'what': 'features'}
    matcher_content = get_matcher(cnn, opt_content)
    matcher_content.mode = 'store'
    cnn(preprocess(im_reference)[None,:].type(conf.data_type))

    # Generator network (prior)
    net_input = get_noise(conf.input_depth, conf.input_type, imsize_net)
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
    iteration = 0
    def train_callback():
        global iteration
        generated = net(net_input)[:, :, :imsize, :imsize]
        generated_preprocessed = vgg_preprocess_var(generated)
        cnn(generated_preprocessed)
        total_loss = sum(matcher_content.losses.values())
        total_loss.backward()
        print ('Iteration %05d    Loss %.3f' %
            (iteration, total_loss.item()), '\r', end='')
        if conf.plot and iteration % 200 == 0:
            generated_np = np.clip(torch_to_np(generated), 0, 1)
            plot_image_grid([generated_np], 3, 3, num=1)
            plt.pause(0.001)
        iteration += 1
        return total_loss

    matcher_content.mode = 'match'
    p = get_params('net', net, net_input)
    optimize(conf.optimizer, p, train_callback, conf.lr, conf.num_iter)

    # Final result
    out = net(net_input)[:, :, :imsize, :imsize]
    out_image = torch_to_np(out)
    plot_image_grid([out_image], 3, 3, num=1)    
    Image.fromarray((255 * out_image).astype(np.uint8).transpose(1, 2, 0)).save(output_file)