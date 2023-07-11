import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim
import prediction_utility

parser = argparse.ArgumentParser(
    description = 'This program is for prediction using saved network'
)

parser.add_argument('flower_path', default='./flowers/test/58/image_02743.jpg', nargs='?', action="store", type = str,help="Give the full path of flower like:flowers/test/100/image_07902.jpg" )
parser.add_argument('path_to_checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str,help="Give the name of checkpoint you saved from trianed network")
parser.add_argument('--top_k', default=1, dest="top_k", action="store", type=int,help="input for most likely clases of your given flower image")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json',help="mapper of categories to real names")
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu",help="Use GPU for inference")

args = parser.parse_args()
flower_path = args.flower_path
checkpoint_path = args.path_to_checkpoint
most_likely_clases= args.top_k
mapper=args.category_names
device_type = args.gpu



def main():
    # This just for defining model architecture before loading our check point
    
     model=prediction_utility.load_checkpoint(checkpoint_path)
     with open(mapper, 'r') as json_file:
            cat_to_name = json.load(json_file)

        # Inference for classification
     classes = prediction_utility.sanity_checking(flower_path,cat_to_name,model,most_likely_clases)
     
     print("The prediction is:{}".format(classes))

    
if __name__== "__main__":
    main()