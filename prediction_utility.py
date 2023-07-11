import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
import PIL
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim

## Loading the checkpoint
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location ='cpu')
    if checkpoint['input_size']==25088:
       model = models.vgg16(pretrained=True)
       model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_layer']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_layer'], 102),
                                     nn.LogSoftmax(dim=1))
    elif checkpoint['input_size']==1024:
         model = models.densenet121(pretrained=True)
         model.classifier = nn.Sequential(nn.Linear(1024,checkpoint['hidden_layer']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_layer'], 102),
                                     nn.LogSoftmax(dim=1))
        
        
        
    input_size =checkpoint['input_size'],
    out_size=checkpoint['output_size'],
    hidden_layer=checkpoint['hidden_layer'],
    optimizer_state=checkpoint['optimizer_state'],
    numer_of_epochs=checkpoint['epochs'],
    model.class_to_idx=checkpoint['model_class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

## Image Preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # TODO: Process a PIL image for use in a PyTorch model
    img=PIL.Image.open(image)
    processed_img=transform(img)
    
    return processed_img

## Class Prediction
def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.cpu()
    image = process_image(image_path)
    # resize the tensor (add dimension for batch)
    image = image.unsqueeze(0)
    model.eval()
    
    output = model(image)
    
    
    ps = torch.exp(output)
    top_probs,top_idx = ps.topk(topk,dim=1)
    
    idx_to_class={}
    for key,values in model.class_to_idx.items():
        idx_to_class[values]=key
    top_idx=top_idx[0].tolist()
    top_probs=top_probs[0].tolist()
    top_classes=[]
    for i in top_idx:
        top_classes.append(idx_to_class[i])

    
    return top_probs, top_classes

def sanity_checking(image_path,num_to_label,model,top_k):
    probs,classes=predict(image_path, model, top_k)
    img_filename = image_path.split('/')[-2]
    if len(classes)==1:
        flower_name = num_to_label[img_filename]
   
    
    else:
        flower_name=[]
        for i in classes:
            flower_name.append(num_to_label[str(i)])


    
    
    return flower_name

    