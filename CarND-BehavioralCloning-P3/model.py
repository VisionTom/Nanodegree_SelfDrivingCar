
# coding: utf-8

# ## Imports

# In[ ]:

#Imports
import ipdb
#%pdb

import pandas as pd
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.model_selection import train_test_split

import csv, random, numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift

from keras.models import model_from_json

get_ipython().magic('matplotlib inline')


# ## Constants

# In[ ]:

#Constants

#Paths
PATH_TRAIN_FOLDER = 'Training_Data/Udacity_Training_Data/'
PATH_VALIDATION1 = 'Training_Data/VAL_TRACK1/'
PATH_VALIDATION2 = 'Training_Data/VAL_TRACK2/'
FILENAME_CSV = 'driving_log.csv'

#Image 
IMAGE_CUT_TOP_HEIGHT = 55
IMAGE_CUT_DOWN_HEIGHT = 25
IMAGE_RESIZE_WIDTH = 64
IMAGE_RESIZE_HEIGHT = 64

#Camera
CAMERA_LEFT_RIGHT_OFFSET = 0.2

#Chances for Augmentation
CHANCES_SHIFT = 0.5
CHANCES_FLIP = 0.5
CHANCES_DARKEN = 0.5
BRIGHTNESS_RANGE = 0.3

#Further Parameters
SPEED_MINIMUM = 20


# ## Read CSVs

# In[ ]:

#Read CSV
def read_csv(path):
    X, y = [], [] 
    
    csv = pd.read_csv(path)
    
    #Throw away slow instances
    csv = csv[(csv['speed']>SPEED_MINIMUM)]

    for index, row in csv.iterrows():
        #center
        X.append(row['center'].strip())
        y.append(row['steering'])
        #left
        X.append(row['left'].strip())
        y.append(row['steering']+CAMERA_LEFT_RIGHT_OFFSET)
        #right
        X.append(row['right'].strip())
        y.append(row['steering']-CAMERA_LEFT_RIGHT_OFFSET)
        
    return X,y


# ## Read Images

# In[ ]:

#Read Images

def resize_and_normalize(img):
    #Cutting the Top and the Bottom of the image
    img_cut = img[IMAGE_CUT_TOP_HEIGHT:160-IMAGE_CUT_DOWN_HEIGHT, :, :]

    #Resize to smaller image size
    img_resize = cv2.resize(img_cut, (IMAGE_RESIZE_WIDTH, IMAGE_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
            
    #Normalizing to a range of -0.5 to +0.5
    img_norm = (img_resize / 255. - .5).astype(np.float32)

    return img_norm


# ## Augmentation

# In[ ]:

def augmentation(path, steering, validation):    

    #Load
    image = cv2.imread(path)
    
    #Augment
    if not validation:
        #Darken
        if random.random() < CHANCES_DARKEN:
            image = random_darken(image)

        #Shift
        if random.random() < CHANCES_SHIFT:
            image = random_shift(image, 0, 0.2, 0, 1, 2)

        #Flip
        if random.random() < CHANCES_FLIP:
            image = flip_axis(image,1)
            steering = steering * -1    
            
    #Resize
    image = resize_and_normalize(image)

    
    return image, steering
    
def random_darken(image):
    
    w = image.shape[0]
    h = image.shape[1]
    
    # Convert the image to HSV
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    
    # Create a random Box
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)
    for i in range(x1, x2):
        for j in range(y1, y2):
            temp[i,j, 2] = temp[i, j, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)


# ## Model

# In[ ]:

#Model
def model(load, shape, checkpoint=None):
    
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, activation='elu', input_shape=shape))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 5, 5, activation='elu'))
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(32, 5, 5, activation='elu'))
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D())
        
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    
    model.add(Dense(1024, activation='elu'))
    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(10, activation='elu'))

    
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model


# In[ ]:

#NVIDIA-Model
def NVIDIA_model(load, shape, checkpoint=None):

    model = Sequential()
    model.add(Convolution2D(24, 5, 5,subsample=(2, 2), activation='elu', input_shape=shape))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3,activation='elu'))
    model.add(Convolution2D(64, 3, 3,activation='elu'))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer="adam")
    
    return model


# Based on the [NVIDIA-Architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):
# ![image alt >](res/nvidia.png)

# In[ ]:

def _generator(batch_size, X, y, path, validation=False):    
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            
            sample_index = random.randint(0, len(X) - 1)
            sa = y[sample_index]       
            
            image, sa = augmentation(path+X[sample_index], sa, validation)
            batch_X.append(image)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)


# In[ ]:

def save_model(net):
    net.save('model.h5')
    
    json_string = net.to_json()
    with open('model.json', 'w') as outfile:
        outfile.write(json_string)


# In[ ]:

def train(net,X, y, path):

    net.fit_generator(_generator(256, X, y, path), samples_per_epoch=21990, nb_epoch=8)
    save_model(net)
    return net


# In[ ]:

def evaluate(net,X, y, path):
    return net.evaluate_generator(_generator(256, X, y, path, validation=True), val_samples=4000)


# In[ ]:

def learn_drivinig():
    #Build model
    net = model(load=False, shape=(IMAGE_RESIZE_WIDTH, IMAGE_RESIZE_HEIGHT, 3))
    
    #Read data and train test split them
    X,y = read_csv(PATH_TRAIN_FOLDER+FILENAME_CSV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #Training
    train(net, X_train, y_train, PATH_TRAIN_FOLDER)
    
    #Evaluation - Validation
    loss = evaluate(net, X_test, y_test, PATH_TRAIN_FOLDER)
    print("Evaluation - Validation: {}".format(loss))
    
    #Evaluation - Testset #1
    X_test,y_test = read_csv(PATH_VALIDATION1+FILENAME_CSV)
    loss = evaluate(net, X_test, y_test, PATH_VALIDATION1)
    print("Evaluation - Test#1: {}".format(loss))
        
    #Evaluation - Testset#2
    X_test,y_test = read_csv(PATH_VALIDATION2+FILENAME_CSV)
    loss = evaluate(net, X_test, y_test, PATH_VALIDATION2)
    print("Evaluation - Test#2: {}".format(loss))


# In[ ]:

def parameter_checker():
    print("*------- STANDARD ---------*")
    learn_drivinig()
    
    #Camera
    CAMERA_LEFT_RIGHT_OFFSET = 0.1
    print("*------- CAMERA_LEFT_RIGHT_OFFSET = 0.1 ---------*")
    learn_drivinig()
    
    #Camera
    CAMERA_LEFT_RIGHT_OFFSET = 0.3
    print("*------- CAMERA_LEFT_RIGHT_OFFSET = 0.3 ---------*")
    learn_drivinig()
    
    #resest
    CAMERA_LEFT_RIGHT_OFFSET = 0.2
    
    #Image 
    IMAGE_CUT_TOP_HEIGHT = 65
    IMAGE_CUT_DOWN_HEIGHT = 35 
    print("*------- IMAGE_CUT_TOP_HEIGHT = 65 \nIMAGE_CUT_DOWN_HEIGHT = 35  ---------*")
    learn_drivinig()
    
    #Image 
    IMAGE_CUT_TOP_HEIGHT = 45
    IMAGE_CUT_DOWN_HEIGHT = 15 
    print("*------- IMAGE_CUT_TOP_HEIGHT = 45 \nIMAGE_CUT_DOWN_HEIGHT = 15 ---------*")
    learn_drivinig()  
    
    #reset
    IMAGE_CUT_TOP_HEIGHT = 55
    IMAGE_CUT_DOWN_HEIGHT = 25 
    
    IMAGE_RESIZE_WIDTH = 64
    IMAGE_RESIZE_HEIGHT = 64
    print("*------- IMAGE_RESIZE_WIDTH = 64 \nIMAGE_RESIZE_HEIGHT = 64 ---------*")
    learn_drivinig()  
    
    IMAGE_RESIZE_WIDTH = 32
    IMAGE_RESIZE_HEIGHT = 32
    print("*------- IMAGE_RESIZE_WIDTH = 32 \nIMAGE_RESIZE_HEIGHT = 32 ---------*")
    learn_drivinig()  
    
    #reset
    IMAGE_RESIZE_WIDTH = 100
    IMAGE_RESIZE_HEIGHT = 100
    
    #NoAugmentation
    CHANCES_SHIFT = 0.0
    CHANCES_FLIP = 0.0
    CHANCES_DARKEN = 0.0
    print("*------- No Augmentation ---------*")
    learn_drivinig()  
    
    #reset
    CHANCES_SHIFT = 0.5
    CHANCES_FLIP = 0.5
    CHANCES_DARKEN = 0.5


# In[ ]:

if __name__ == '__main__':   
    #parameter_checker()
    learn_drivinig()


# In[ ]:

with open('model.json', 'r') as jfile:
          model = model_from_json(jfile.read())

model.summary()



# In[ ]:

n_model = model(load=False, shape=(100,100,3))
n_model.summary()

