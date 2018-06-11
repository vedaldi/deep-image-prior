import os
import sys
import glob
from PIL import Image
import inverter
from utils.feature_inversion_utils import View # For Pickle

prefix = 'data/blue'
group_size = 1

inverter.conf.plot = False
inverter.conf.cuda = None
#inverter.conf.num_iter = 3

# What to do based on sysarg
if len(sys.argv) > 1:
    case = int(sys.argv[1])
    if case == 1:
        inverter.conf.layer_to_invert = 'conv5'
        inverter.conf.cuda = '0'
    elif case == 2:
        inverter.conf.layer_to_invert = 'fc6'
        inverter.conf.cuda = '1'
    elif case == 3:
        inverter.conf.layer_to_invert = 'fc8'
        inverter.conf.cuda = '2'

if len(sys.argv) > 2:
    inverter.conf.cuda = sys.argv[2]

if len(sys.argv) > 3:
    in_order = bool(int(sys.argv[3]))

print(in_order)

# Create destination folder
x1_folder = os.path.join(prefix, inverter.conf.layer_to_invert)
x0_folder = os.path.join(prefix, 'x0')
inverter.xmkdir(x1_folder)
inverter.xmkdir(x0_folder)

# Load network
cnn = inverter.load_net(inverter.conf)

# Remove layers we do not need
inverter.slice_net(inverter.conf, cnn)

# Running jobs
def run(job):
    ims = [Image.open(path) for path, _, _ in job]
    x1s, x0s = inverter.invert(inverter.conf, cnn, ims)
    for x0, x0p in zip(x0s, [x for _, x, _ in job]):
        x0.save(x0p)
    for x1, x1p in zip(x1s, [x for _, _, x in job]):
        x1.save(x1p)

# Load and normalise image
jobs = []
paths = glob.glob(os.path.join(prefix, "*.jpg"))
if in_order:
    paths = list(sorted(paths))
for path in paths:
    x0_path = os.path.join(x0_folder, os.path.basename(path))
    x1_path = os.path.join(x1_folder, os.path.basename(path))
    print(path, x0_path, x1_path)
    if os.path.exists(x1_path):
        print("Skipping because it already exists.")
        continue
    jobs.append((path, x0_path, x1_path))
    if len(jobs) == group_size:
        run(jobs)
        jobs = []
run(jobs)

