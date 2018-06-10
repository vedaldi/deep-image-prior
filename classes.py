import os
from PIL import Image
import PIL
import maximizer 
from utils.feature_inversion_utils import View # For Pickle
from imnet_list import imnet_classes

class_names = ["black swan", "cheeseburger", "goose", "coffee mug", 
    "vending machine", "tree frog", "volcano"]
class_names = [x for _, x in imnet_classes.items()]

maximizer.xmkdir('data/maxim2')

for class_name in reversed(class_names):
    maximizer.conf.layer_to_maximize = "fc8"
    neuron = maximizer.get_neuron_for_class(class_name)

    # Load network
    cnn = maximizer.load_net(maximizer.conf)
    maximizer.slice_net(maximizer.conf, cnn)

    # Maximize
    x0 = maximizer.maximize(maximizer.conf, cnn, neuron)
    x0.save(os.path.join('data/maxim', class_name + ".png"))
