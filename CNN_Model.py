import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as model
from tensorflow.keras.models import load_model
import string
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import CNN_Model,Image_Segmentation

class Model:
    def train_model():
        inputs=layers.Input(x_input[0].shape)
        c1=layers.Conv2D(32,(3,3),activation='relu',padding='valid')(inputs)
        p1=layers.MaxPool2D((2,2))(c1)
        c2=layers.Conv2D(16,(3,3),activation='relu',padding='valid')(p1)
        p2=layers.MaxPool2D((2,2))(c2)
        f=layers.Flatten()(p2)
        outputs=layers.Dense(36,activation=tf.nn.softmax)(f)    
        cnn_model=model.Model(inputs,outputs)
        return cnn_model
    
    def predict_output(self,png_images,segmentor):
        encoded_alphabets={10+i:alphabet for i,alphabet in enumerate(string.ascii_uppercase)}
        number_plate=[]
        for image in png_images:
            digits,plate=segmentor.grab_test_info(image)
            if len(digits)!=7:            
                digits,plate=segmentor.grab_bad_data_info(image)
                if len(digits)>5 and len(digits)<=7:
                    plate=''
                    for digit in digits:
                        x_array=np.array(digit)/255
                        test_input=x_array.reshape(1,40,15,1)
                        model=load_model('detection.h5')
                        prediction=model.predict(test_input)
                        if np.argmax(prediction)>=10:
                            plate+=str(encoded_alphabets[np.argmax(prediction)])
                        else:
                            plate+=str(np.argmax(prediction))
                    number_plate.append(plate)                
                else:
                    number_plate.append('Sorry..!! Could not detect')

            else:
                plate=''
                for digit in digits:
                    x_array=np.array(digit)/255
                    test_input=x_array.reshape(1,40,15,1)
                    model=load_model('detection.h5')
                    prediction=model.predict(test_input)
                    if np.argmax(prediction)>=10:
                        plate+=str(encoded_alphabets[np.argmax(prediction)])
                    else:
                        plate+=str(np.argmax(prediction))
                number_plate.append(plate)
        return number_plate

