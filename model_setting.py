# Contains functions and classes relating to the model
import torch
import torchvision
import torchvision.models as models
import torchvision.models as models
from torch import nn, optim



# Building and training the classifier
def definning_network(model_architecture='vgg16',hidden_units=256, learning_rate=0.001, device='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        model
    elif model_architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        model
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict  
    if model_architecture=='vgg16':
        input_size=25088
    else:
        input_size=1024

    model.classifier = nn.Sequential(
         nn.Linear(input_size , hidden_units),
         nn.ReLU(),
         nn.Dropout(0.2),
         nn.Linear(hidden_units, 102),
         nn.LogSoftmax(dim=1)
     )
    print(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.to(device);
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion,optimizer
