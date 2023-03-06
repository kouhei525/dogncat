import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
class_name = ['abyssinian',
 'american_bulldog',
 'american_pit_bull_terrier',
 'basset_hound',
 'beagle',
 'bengal',
 'birman',
 'bombay',
 'boxer',
 'british_shorthair',
 'chihuahua',
 'egyptian_mau',
 'english_cocker_spaniel',
 'english_setter',
 'german_shorthaired',
 'great_pyrenees',
 'havanese',
 'japanese_chin',
 'keeshond',
 'leonberger',
 'maine_coon',
 'miniature_pinscher',
 'newfoundland',
 'persian',
 'pomeranian',
 'pug',
 'ragdoll',
 'russian_blue',
 'saint_bernard',
 'samoyed',
 'scottish_terrier',
 'shiba_inu',
 'siamese',
 'sphynx',
 'staffordshire_bull_terrier',
 'wheaten_terrier',
 'yorkshire_terrier']

model = torchvision.models.resnet18(weights=True)
model.fc = nn.Linear(512, 37)
param = torch.load("pet_is.pth", map_location=torch.device('cpu'))
model.load_state_dict(param)

def predict(path):
    #im = Image.open(path)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(IMG_MEAN, IMG_STD)])
    x = transform(path)
    model.eval()
    imx = torch.unsqueeze(x, 0)
    outputs = model(imx)
    #y = torch.argmax(outputs)
    dog = torch.squeeze(outputs)
    s, idx = torch.sort(dog, descending=True)
    a, b, c = idx[0:3].tolist()
    return class_name[a],class_name[b],class_name[c]

#ans = predict("main_53582_8e4bf_detail.jpg")
#print(ans)