#import torch
#import torchvision
import numpy as np

import argparse

import sys
import os
from pathlib import Path

from PIL import Image

#import torch.nn as nn
#import torch.nn.functional as F

from sklearn.svm import LinearSVC, SVC
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle



#what's the right way to do this? we could have the user give us a repository of files, and then do classification on all of them
# in that case, we need the user to provide a path to the files, and a choice of what model to run (svm or rfc)
#and what would we return? Probably a dictionary, between filename and predicted label

parser = argparse.ArgumentParser(description='Run inference on a folder of images')

parser.add_argument("data_path", help="the path to a folder full of images you want to classify with the model")
parser.add_argument("model_type", help="'svm' or 'rfc'")

args = parser.parse_args()

arg_dict = vars(args)

#load and process the data
images_path = Path(arg_dict['data_path'])
images_list = sorted(entry for entry in images_path.iterdir() if entry.is_file())
img_arrays_list = []
for item in images_list:
    if item.name == '.DS_Store': #handling this exception
        continue
    image = Image.open(item).convert('L')
    img_data = np.asarray(image)
    img_arrays_list.append((img_data))

img_data_fourier = np.fft.fft2(img_arrays_list)
img_data_fourier = np.absolute(img_data_fourier).reshape(len(img_data_fourier),600*800)

###load the model
loaded_model = pickle.load(open(f"./successful_{arg_dict['model_type']}_1.sav", 'rb'))


#predict with the model

predictions = loaded_model.predict(img_data_fourier)
print(predictions)



