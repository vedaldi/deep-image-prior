import requests
import os
from PIL import Image
import PIL
import inverter 
from utils.feature_inversion_utils import View # For Pickle

prefix = 'data/sequence'
inverter.xmkdir(prefix)
inverter.conf.cuda = '1'

# Load demo image
url = 'https://upload.wikimedia.org/wikipedia/commons/2/27/Baby_ginger_monkey.jpg'
im = Image.open(requests.get(url, stream=True).raw)
w, h = im.size
im = im.resize((int(w/h*256),256), PIL.Image.LANCZOS)

# Load network
cnn = inverter.load_net(inverter.conf)

# Go
for k in reversed(list(cnn._modules.keys())):
    print("Slice", k)
    inverter.conf.layer_to_invert = k    
    #inverter.conf.num_iter = 32  
    inverter.slice_net(inverter.conf, cnn)
    x1, x0 = inverter.invert(inverter.conf, cnn, im)
    x0.save(os.path.join(prefix, 'x0.png'))
    x1.save(os.path.join(prefix, k +'.png'))
