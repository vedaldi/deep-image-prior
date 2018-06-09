import os
from PIL import Image
import PIL
import maximizer 
from utils.feature_inversion_utils import View # For Pickle

maximizer.conf.layer_to_maximize = "fc8"
neuron = maximizer.get_neuron_for_class('black swan')

# Load network
cnn = maximizer.load_net(maximizer.conf)
maximizer.slice_net(maximizer.conf, cnn)

# Maximize
x0 = maximizer.maximize(maximizer.conf, cnn, neuron)