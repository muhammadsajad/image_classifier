# Contains utility functions like loading data and preprocessing images
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict


def load_data(root = "./flowers"):
    
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
  # Define your transforms for the training, validation, and testing sets
# data_transforms = 
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 

    validation_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    testing_transforms=transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    # image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data=datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data=datasets.ImageFolder(test_dir, transform=testing_transforms)

    image_datasets = [train_data, validation_data, test_data]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader=torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


    return trainloader, validationloader, testloader, train_data