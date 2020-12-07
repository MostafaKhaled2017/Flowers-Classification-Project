import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import json

import argparse

from collections import OrderedDict

import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import glob, os
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Model parameters')
parser.add_argument('data_directory', help='data directory')
parser.add_argument('--save_dir', help='directory to save the neural network.')
parser.add_argument('--arch', help='the available models. Options are:vgg,densenet')
parser.add_argument('--learning_rate', help='learning rate')
parser.add_argument('--hidden_units', help='the number of hidden units')
parser.add_argument('--epochs', help='epochs')
parser.add_argument('--gpu',action='store_true', help='gpu')
args = parser.parse_args()

if args.arch not in ('vgg','densenet',None):
    raise Exception('Please choose vgg or densenet')
if (args.gpu and not torch.cuda.is_available()):
    raise Exception("You do not have GPU")
if(not os.path.isdir(args.data_directory)):
    raise Exception('Directory does not exist!')
    data_dir = os.listdir(args.data_directory)


data_dir = 'flowers'
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if (args.arch is None):
    arch_type = 'vgg'
else:
    arch_type = args.arch
if(args.arch is 'densenet'):
    model = models.densenet121(pretrained=True)
    input_node=1024
else:
   model = models.vgg19(pretrained=True)
   input_node=25088

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
    if (args.learning_rate is None):
        lr = 0.001
    else:
        lr = float(args.learning_rate)
    if (args.epochs is None):
        epochs = 10
    else:
        epochs = int(args.epochs)
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
   
if (args.hidden_units is None):
    hidden_units = 4096
else:
    hidden_units = int(args.hidden_units)

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_node, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

        
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model.to(device)

print_every = 75
steps = 0
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(train_loader):
        steps += 1
        
        optimizer.zero_grad()
        
        images, labels = images.to(device), labels.to(device)
        
        # Forward and backward passes
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validation_loader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader) *100))
            
            running_loss = 0
            
            #To make training back on
            model.train()
            
#save checkpoint
print('saving checkpoint')

if (args.save_dir is None):
    save_dir = 'check.pth'
else:
    save_dir = args.save_dir
 
checkpoint = {'optimizer' : optimizer,
              'classifier' : model.classifier,
              'model' : model,
              'class_to_idx' : train_data.class_to_idx,
              'optimizer_dict': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'arch': 'densenet121'}

torch.save(checkpoint, save_dir)
   
print('The Model is ready now')