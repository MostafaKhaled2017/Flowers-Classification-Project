import argparse 
import torch 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

parser = argparse.ArgumentParser(description='predicting the class of the image')
parser.add_argument('image_input', help='image file')
parser.add_argument('model_checkpoint', help='model')
parser.add_argument('--top_k', help='how many prediction categories to show')
parser.add_argument('--category_names', help='file for category names')
parser.add_argument('--gpu', action='store_true', help='gpu option')
args = parser.parse_args()

if (args.gpu and not torch.cuda.is_available()):
    raise Exception("--gpu option enabled...but no GPU detected")
if (args.top_k is None):
    top_k = 5
else:
    top_k = int(args.top_k)

image_path = args.image_input
   
    
if (args.category_names is not None):
    f = args.category_names 
    jfile = json.loads(open(f).read())
else:
    jfile = None

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    
    if(checkpoint['arch'] == 'densenet'):
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.arch = checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_dict'])

    
    
    return model

loaded_model = load_checkpoint(args.model_checkpoint)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])
    
    img = Image.open(image)
    np_img = img_transforms(img).numpy()

    return np_img

np_image = process_image('flowers/test/1/image_06743.jpg')

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    if (args.gpu) and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    image = process_image(image_path)
    
    if device == 'cuda':
        image_tensor = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    
    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    model.to(device)
    output = model(image_tensor)
    ps = torch.exp(output)

    probs, indices = ps.topk(topk, dim=1)
    
    idx_to_class = {value: key for key,value in model.class_to_idx.items()}

    prob = [p.item() for p in probs[0].data]

    classes = [idx_to_class[i.item()] for i in indices[0].data]
    return prob, classes, image  
    
    
    # TODO: Implement the code to predict the class from an image file
probs, classes, image = predict(image_path , loaded_model, top_k)

cat_file = jfile



if cat_file is None:
    labels = classes
else:
    labels = [cat_file[str(index)] for index in classes]
probability = probs
    
i=0 
while i < len(labels):
    print("{} with a probability of {}".format(labels[i], probability[i]*100))
    i += 1