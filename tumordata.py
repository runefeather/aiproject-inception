# Based on cifar10.py
# Sorts data into required sections
########################################################################
import numpy as np
from PIL import Image
import os
import download
from dataset import one_hot_encoded
from helpers import downSampling
########################################################################
# data path for images
global data_path
global trdir
global tedir
global valdir


# Various constants for the size of the images.
# Width and height of each image.
img_width = 460
img_height = 700

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_width * img_height * num_channels

# Number of classes.
num_classes = 2

# Total number of images in the training, test and validation sets.
# This is used to pre-allocate arrays for efficiency.

global _num_images_train 
global _num_images_test 
global _num_images_validation

# Dictionary for test, training and validation data
TRAIN_DICTIONARY = {}
TEST_DICTIONARY = {}
VALIDATION_DICTIONARY = {}

def start():
    global _num_images_train 
    global _num_images_test 
    global _num_images_validation
    
    _num_images_train =  len(os.listdir(trdir))
    _num_images_test =  len(os.listdir(tedir))
    _num_images_validation =  len(os.listdir(valdir))

    tef = open(os.path.join(data_path, "testdata.txt"))
    trf = open(os.path.join(data_path, "traindata.txt"))
    vaf = open(os.path.join(data_path, "valdata.txt"))

    for line in tef:
        # print(line.split(" "))
        # print(line.split(" ")[0].split("/")[-1].strip(), line.split(" ")[1].strip())
        TEST_DICTIONARY[line.split(" ")[0].split("/")[-1].strip()] = line.split(" ")[1].strip()

    for line in trf:
        # print(line.split(" "))
        # print(line.split(" ")[0].split("/")[-1].strip(), line.split(" ")[1].strip())
        TRAIN_DICTIONARY[line.split(" ")[0].split("/")[-1].strip()] = line.split(" ")[1].strip()

    for line in tef:
        # print(line.split(" "))
        # print(line.split(" ")[0].split("/")[-1].strip(), line.split(" ")[1].strip())
        VALIDATION_DICTIONARY[line.split(" ")[0].split("/")[-1].strip()] = line.split(" ")[1].strip()

"""
There are two classes - benign and malignant
Returns a list with the names. Example: names[0] is the name
associated with class-number 0.
"""
def load_class_names():
    names = ['benign', 'malignant']
    return names

def add_data_path(splitname):
    global data_path
    global trdir
    global tedir
    global valdir
    data_path = "/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitname
    trdir = '/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/' + splitname + '/training'
    tedir = '/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/' + splitname + '/testing'
    valdir = '/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/' + splitname + '/validation'

# loads testing data
def load_testing_data():
    images = np.zeros(shape=[_num_images_test, img_width, img_height, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_test], dtype=int)
    a = TEST_DICTIONARY.keys()
    for i in range(0, _num_images_test):
        f = os.listdir(os.path.join(data_path, 'testing'))[i]
        if f in a:
            img = Image.open(os.path.join(data_path, 'testing', f))
            arr = np.array(img)
            if(len(arr[0]) == 460): images[i] = arr
            cls[i] = TEST_DICTIONARY[f]
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_training_data():
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_width, img_height, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    a = TRAIN_DICTIONARY.keys()
    for i in range(0, _num_images_train):
        f = os.listdir(os.path.join(data_path, 'training'))[i]
        if f in a:
            img = downSampling(os.path.join(data_path, 'training', f), 300, 300)
            arr = np.array(img)
            if(len(arr[0]) == 460): images[i] = arr
            cls[i] = TRAIN_DICTIONARY[f]
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_validation_data():
    images = np.zeros(shape=[_num_images_validation, img_width, img_height, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_validation], dtype=int)
    a = VALIDATION_DICTIONARY.keys()
    for i in range(0, _num_images_validation):
        f = os.listdir(os.path.join(data_path, 'validation'))[i]
        if f in a:
            img = Image.open(os.path.join(data_path, 'validation', f))
            arr = np.array(img)
            if(len(arr[0]) == 460): images[i] = arr
            cls[i] = VALIDATION_DICTIONARY[f]
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)




