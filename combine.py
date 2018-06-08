import sys
import os
import cv2
import PIL
import glob
import numpy as np

prefix = 'data/blue'
output = 'data/blue.mp4'

def read_frame(file):
    base = os.path.basename(file)
    ims = []
    for quad in ["x0", "conv5", "fc6", "fc8"]:
        path = os.path.join(prefix, quad, base)
        #print(path)
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            ims.append(im)
        else:
            ims.append(np.zeros_like(ims[-1]))
    return np.concatenate(
            (np.concatenate(ims[0:2], axis=1),
             np.concatenate(ims[2:4], axis=1)), axis=0)

files = sorted(glob.glob(os.path.join(prefix, "x0", "*.jpg")))
frame = read_frame(files[0])

fourcc = cv2.VideoWriter_fourcc(*'x264')#(*'x264') #(*'mp4v')
video = cv2.VideoWriter(output, fourcc, 20.0, frame.shape[:2])
for file in files:
    frame = read_frame(file)
    video.write(frame)
    print('.', end='')
video.release()
