import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim
import data_loader
import model_setting

parser = argparse.ArgumentParser(
    description = 'This program is used for training by giving the name of pretrained network and hyperparametrs'
)
arch = {"vgg16":25088,
        "densenet121":1024}
parser.add_argument('data_dir', action="store",default="./flowers",help="Giving the name of directory of dataset like flowers")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth",help="Give name of directory where you want to save your model check point")
parser.add_argument('--arch', action="store",default="vgg16",choices=['vgg16', 'densenet121'],help="Give the name of pretrianed network like vgg16,vgg19 or desnet etc")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001,help="learning rate for gradient steps")
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=256,help="hidden units for classifier")
parser.add_argument('--epochs', action="store", default=2, type=int,help="Numer epochs means how many time you want to train your model on data")
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu",help="Use gpu for trainning if availile")

in_args = parser.parse_args()
data_dir_path = in_args.data_dir
path_to_save_chckpint=in_args.save_dir
learning_rate=in_args.learning_rate
model_architecture = in_args.arch
hidden_units = in_args.hidden_units
device_type =in_args.gpu
number_epochs =in_args.epochs

if device_type == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
def main():
     
     # Load the data and preprocssed it
    train_loader, validation_loader, test_loader, train_data = data_loader.load_data(data_dir_path)
    model, criterion,optimizer = model_setting.definning_network(model_architecture,hidden_units,learning_rate,device_type)
     # Train Model
    epochs = number_epochs
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validation_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validation_loader):.3f}")
                running_loss = 0
                model.train()
    
    ## Save the checkpoint
                    
    model.class_to_idx = train_data.class_to_idx
    if model_architecture == 'vgg16':
        input_size=25088
    elif model_architecture == 'densenet121':
        input_size=1024
                    
    checkpoint = {'input_size':input_size,
              'output_size': 102,
              'hidden_layer':hidden_units,
              'state_dict': model.state_dict(),
             'optimizer_state':optimizer.state_dict(),
             'epochs':number_epochs,
             'model_class_to_idx':model.class_to_idx}

    torch.save(checkpoint, path_to_save_chckpint)
                    
   
    print("Trainnig process dome succesfully and check point saved")
if __name__== "__main__":
    main()

