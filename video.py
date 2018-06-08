import os
import sys
import glob
from PIL import Image
import inverter
from utils.feature_inversion_utils import View # For Pickle

prefix = 'data/blue'

# What to do based on sysarg
if len(sys.argv) > 1:
    case = int(sys.argv[1])
    if case == 1:
        inverter.conf.layer_to_invert = 'fc6'
        inverter.conf.cuda = '0'
    elif case == 2:
        inverter.conf.layer_to_invert = 'conv5'
        inverter.conf.cuda = '2'
    elif case == 3:
        inverter.conf.layer_to_invert = 'fc8'
        inverter.conf.cuda = '3'

# Create destination folder
x0_folder = os.path.join(prefix, inverter.conf.layer_to_invert)
input_folder = os.path.join(prefix, 'x0')
inverter.xmkdir(x0_folder)
inverter.xmkdir(input_folder)

# Load network
cnn = inverter.load_net(inverter.conf)

# Remove layers we do not need
inverter.slice_net(inverter.conf, cnn)

# Load and normalise image
for path in glob.glob(os.path.join(prefix, "*.jpg")):
    x0_path = os.path.join(input_folder, os.path.basename(path))
    x1_path = os.path.join(x0_folder, os.path.basename(path))
    print(x0_path, x1_path)
    if os.path.exists(x1_path):
        print("Skipping because it already exists.")
        continue
    im = Image.open(path)
    x1, x0 = inverter.invert(inverter.conf, cnn, im)
    x0.save(x0_path)
    x1.save(x1_path)
