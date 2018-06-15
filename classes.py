import os
from PIL import Image
import PIL
import maximizer 
from utils.feature_inversion_utils import View # For Pickle
from imnet_list import imnet_classes

prefix = 'data/maxim'
class_names = [x for _, x in imnet_classes.items()]

maximizer.xmkdir(prefix)
#maximizer.conf.cuda = None
#maximizer.conf.pretrained_net = 'alexnet_torch'

for class_name in reversed(class_names):
    out_path = os.path.join(prefix, class_name + ".png")
    print(out_path)
    if os.path.exists(out_path):
        print("Skipping", out_path)
        continue
    maximizer.conf.layer_to_maximize = "fc8"

    neuron = maximizer.get_neuron_for_class(class_name)

    # Load network
    cnn = maximizer.load_net(maximizer.conf)
    maximizer.slice_net(maximizer.conf, cnn)

    # Maximize
    x0 = maximizer.maximize(maximizer.conf, cnn, neuron)
    x0.save(out_path)
