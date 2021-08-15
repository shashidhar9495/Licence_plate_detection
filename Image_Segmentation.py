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



### Filtering and Segmentation

class Segmentation:
    
    def sharpen(self,original_image):
        kernal=kernel = np.array([[0,0,0], 
                       [0,1,0],
                       [0,0,0]])
        sharpened_image =cv2.filter2D(original_image,ddepth=-1,kernel=kernal)
        return sharpened_image


    def thresholding(self,original_image,gray_image):
        laplacian_var = cv2.Laplacian(original_image, cv2.CV_64F).var()
        if laplacian_var<300:
            blur=cv2.GaussianBlur(original_image,(3,3),4)
            blur=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            threshold=cv2.adaptiveThreshold(blur,blur.max(),cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,3)
            return threshold
        if gray_image.max()<250:
            sharp_image=self.sharpen(original_image)
            grayed_image=cv2.cvtColor(sharp_image,cv2.COLOR_BGR2GRAY)
            _,threshold=cv2.threshold(grayed_image,grayed_image.max()/1.5,grayed_image.max(),cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,1)
            return threshold
        else:
            blur=cv2.GaussianBlur(gray_image,(5,5),2)
            _,threshold=cv2.threshold(gray_image,gray_image.max()/2,gray_image.max(),cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,1)
            return threshold

    def get_contour(self,original_image,gray_image):
        thr_img=self.thresholding(original_image,gray_image)
        contours,_=cv2.findContours(thr_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)
        sorted_ctrs = sorted(contours, key=lambda contours: cv2.boundingRect(contours)[0])
        return sorted_ctrs
    
    def draw_contour(self,original_image,gray_image):
        copy=original_image.copy()
        thr_img=self.thresholding(original_image,gray_image)
        contours,_=cv2.findContours(thr_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contour_img=cv2.drawContours(copy,contours,-1,(0,255,0),1)
        return contour_img,len(contours)
    
    def draw_contour_box(self,original_image,gray_image):
        copy=original_image.copy()
        thr_img=self.thresholding(original_image,gray_image)
        contours,_=cv2.findContours(thr_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
        boxes=[]
        for cntr in contours:
            x,y,w,h=cv2.boundingRect(cntr)
            cv2.rectangle(copy,(x,y),(x+w,y+h),(255,0,0),1)
            boxes.append(original_image[y:y+h,x:x+h])
        return copy,boxes
    
    def draw_boxes(self,original_image,gray_image):
        digits=[]

        contours=self.get_contour(original_image,gray_image)
        copy=original_image.copy()
        copy_gray=gray_image.copy()
        for cntr in contours:
            x,y,w,h=cv2.boundingRect(cntr)
            hw_ratio=h/w
            if h/copy.shape[0]>=0.3 and h/copy.shape[0]<0.5:
                if hw_ratio>1 and hw_ratio<4.3:
                    if w*h>350 and w*h<1500:
                        cv2.rectangle(copy,(x-1,y-1),(x+w+1,y+h+1),(255,0,0),1)
                        masked_image=self.thresholding(original_image,gray_image)
                        curr_num=masked_image[y:y+h,x:x+w]
                        digits.append(cv2.resize(curr_num,(15,40)))
        return digits,copy
    
    def handle_bad_contour(self,original_image,gray_image):
        thr_img=self.thresholding(original_image,gray_image)
        contours,_=cv2.findContours(thr_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(contours, key=lambda contours: cv2.boundingRect(contours)[0])
        return sorted_ctrs

    def bad_data_boxes(self,original_image,gray_image):
        digits=[]

        org=cv2.resize(original_image[10:90,45:220],(250,100))
        blur=cv2.GaussianBlur(org,(3,3),4)
        gry=cv2.resize(gray_image[10:90,45:220],(250,100)) 

        contours=self.handle_bad_contour(blur,gry)

        copy=org.copy()
        copy_gray=gry.copy()
        for cntr in contours:
            x,y,w,h=cv2.boundingRect(cntr)
            hw_ratio=h/w
            if h/copy.shape[0]>=0.3 and h/copy.shape[0]<0.7:
                if hw_ratio>0 and hw_ratio<5:
                    if w*h>450 and w*h<3000:
                        cv2.rectangle(copy,(x-1,y-1),(x+w+1,y+h+1),(255,0,0),1)
                        masked_image=self.thresholding(org,gry)
                        curr_num=masked_image[y:y+h,x:x+w]
                        digits.append(cv2.resize(curr_num,(15,40)))
        return digits,copy
        
        
    def grab_train_info(self,file_name,labels):
        good_digits=[]
        bad_digits=[]
        good_labels=[]
        bad_labels=[]
        boxed_images=[]
        for i,file in enumerate(file_name):
            digits,boxed_image=self.draw_boxes(resized_org_img[file],resized_gray_img[file])
            if len(digits)==7:
                label=list(labels[i])
                for j,digit in enumerate(digits):
                    good_digits.append(digit)
                    good_labels.append(label[j])
            else:
                digits,boxed_image=self.bad_data_boxes(resized_org_img[file],resized_gray_img[file])
                if len(digits)==7:
                    label=list(labels[i])
                    for j,digit in enumerate(digits):
                        good_digits.append(digit)
                        good_labels.append(label[j])
                else:
                    bad_digits.append(file)
                    bad_labels.append(labels[i])
            boxed_images.append(boxed_image)
        return good_digits,good_labels,bad_digits,bad_labels,boxed_images
    
    def grab_test_info(self,png_image):
        original_image=cv2.imread(png_image)
        gray_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
        
        original_resize=cv2.resize(original_image,(250,100))
        gray_resize=cv2.resize(gray_image,(250,100))
        
        digits,boxed_image=self.draw_boxes(original_resize,gray_resize)
        return digits,boxed_image
    
    def grab_bad_data_info(self,png_image):
        original_image=cv2.imread(png_image)
        gray_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
        
        original_resize=cv2.resize(original_image,(250,100))
        gray_resize=cv2.resize(gray_image,(250,100))
        
        digits,boxed_image=self.bad_data_boxes(original_resize,gray_resize)
        return digits,boxed_image
    


