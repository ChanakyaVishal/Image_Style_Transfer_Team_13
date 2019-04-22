import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from dataLoader import Dataset
from wct import *
from torch.utils.serialization import load_lua
import time

parser = argparse.ArgumentParser()

alpha = 1
CONTENT_DIR = 'input/content'
STYLE_DIR = 'input/style'
OUTPUT_DIR = 'output/'
out_size = 512
batch_size = 1
cuda = True
workers = 2


parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7')

args = parser.parse_args()

try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    pass

# Data loading code
dataset = Dataset(CONTENT_DIR, STYLE_DIR,out_size)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)
def styleTransfer(contentImg,styleImg,imname,csF):

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(contentImg)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,csF,alpha)
    Im1 = wct.d1(csF1)

    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.data.cpu().float(),os.path.join(OUTPUT_DIR,imname))
    return