# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:15:00 2018

@author: mrpal
"""

## This is the python script to calibrate the camera and provide camera calibration coefficients

import numpy as np
import cv2
import os


objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

## Make a list of calibration images
#image_dir = "C:\\Users\\mrpal\\Documents\\Projects\\Lane-Tracking\\camera_cal\\"
#images = os.listdir('camera_cal')




def calibrate_camera(image_dir, images):
    index = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(image_dir+images[index])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            storage_name = 'calibration image' +str(idx) + 'with corner ID'+'.jpg'
            #** Turn this on later when you want to save
            
#            if len(glob.glob('./camera_cal/*corner ID*.jpg')) == 0:
#            cv2.imwrite(os.path.join(image_dir , storage_name),img)
            #cv2.waitKey(500)
        index +=1
    cv2.destroyAllWindows()
    return corners, imgpoints, objpoints, gray



#This section calls the camera distortion correction function
def distortion_correct(image_dir,images, mtx, dist):
    print('Try to undistort a sample image:')
    idx = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(image_dir+images[idx])
#        cv2.imshow('img',img)
        #cv2.waitKey(250)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        #cv2.imshow('img',dst)
        #print('showing image {}'.format(idx))
        storage_name = 'Distortion Corrected image # ' +str(idx)+'.jpg'      
#        cv2.imwrite(os.path.join(image_dir , storage_name),dst)
        #cv2.waitKey(250)