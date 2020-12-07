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
    model = checkpoint['model']
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
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image_tensor.unsqueeze_(0)
    image_tensor.to('cuda')
    
    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    model.to('cuda')
    output = model(image_tensor)
    ps = torch.exp(output)

    probs, classes = ps.topk(topk, dim=1)
    
    probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
    results = zip(probs,classes)
    return results, image   
    
    
    # TODO: Implement the code to predict the class from an image file
results, image = predict(image_path , loaded_model, top_k)

cat_file = jfile

i = 0
for prob, classes in results:
    i = i + 1
    prob = str(round(prob,4) * 100.) + '%'
    
    if (cat_file):
        classes = cat_file.get(str(classes),'None')
    else:
        classes = ' class {}'.format(str(classes))
    print("{}.{} ({})".format(i, classes,prob))

