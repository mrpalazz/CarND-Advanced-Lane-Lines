# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:44:55 2017

@author: Mike
"""


import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import os
from scipy import stats
from moviepy.editor import VideoFileClip
from IPython.display import HTML


objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
image_dir = "C:\\Users\\Mike\\Documents\\Self Driving Car nano-degree\\Term 1\\CarND-Advanced-Lane-Lines\\camera_cal\\"
images = os.listdir('camera_cal')




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
def distortion_correct(image_dir,images):
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


#Absolute Sobel Thresholding
#Feed in the images and select a direction as a string, 'x' or 'y' to return the proper gradient

def abs_sobel_image(img, orient, threshold, kernel):    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient =='x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1    
        
    return binary_output

## add in the section for binary transform like in the lecture slide 21



#Sobel gradient magnitude filtering
def sobel_mag_thresh(img, sobel_kernel, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output



#Sobel gradient direction filter
def sobel_dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    #binary_output = binary_output.view(np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    # Return the binary image
    return binary_output


#HLS Color space threshold filter
def color_binary(img, colorspace, color_thresh):

    if colorspace == 'HLS':    
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
    
    binary_output = np.zeros_like(S)
    binary_output[((S > color_thresh [0]) & (S < color_thresh [1]))] = 1
    return binary_output



#combine the thresholds for the color map and the gradient threshold
# send in an image with binary color scheme and binary gradient scheme
def bin_color_gradient(binary_gradient , binary_color):
    
    binary_output = np.zeros_like(binary_gradient)
    binary_output[((binary_gradient == 1) | (binary_color == 1))] = 1
#    polys = np.array([[(350,720),(580,500),(800,500),(1000,720)]], dtype = np.int32)
    polys = np.array([[(350,720),(580,500),(800,500),(900,720)]], dtype = np.int32)

    cv2.fillPoly(binary_output, polys, 0, lineType=8, shift=0) 
    
    
    
    return binary_output

#Function to warp images to birds eye view
def warp(img,source_points, destination_points):
    img_shape = (img.shape[1], img.shape[0])
    src = np.float32(source_points)
    dst = np.float32(destination_points)
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,img_shape, flags = cv2.INTER_LINEAR)
    return warped, M, Minv



def polyfit(warped_image, orig_img, Minv):
#def polyfit(warped_image):
#    print('Initiating line overlay onto binary warped image')
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_image, warped_image, warped_image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window+1)*window_height
        win_y_high = warped_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
   
    
#    plt.figure(figsize = (20,10))
#    plt.imshow(out_img)
#    plt.plot(left_fitx, ploty, color='blue')
#    plt.plot(right_fitx, ploty, color='red')
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)
#    plt.show()
    
    
        # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    
# =============================================================================
#     In this section we calculate the radius of curvature for the warped lines
# =============================================================================
    
        # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
#    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
# =============================================================================
#   Calculate the position from center for the vehicle relative to the left lane 
# =============================================================================
    
    
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_img.shape[1], orig_img.shape[0])) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(newwarp,'Recording: project_video',(10,50), font, 1,(255,0,0),3,cv2.LINE_AA)
    cv2.putText(newwarp,'Road Radius of curvature: {} km'.format(left_curverad/1000),(10,100), font, 1,(255,0,0),3,cv2.LINE_AA)
    

    
# =============================================================================
#     Add the Section for fitting the radius of curvature to the image
# =============================================================================
    
    
    
    vehicle_center = newwarp.shape[1]/2 #assuming that the video feed is from veh center
    y_pixels = np.arange(newwarp.shape[0]-10, newwarp.shape[0]+1)
#    y_pixels = 719
    lx_loc = left_fit_cr[0]*y_pixels**2+left_fit_cr[1]*y_pixels+left_fit_cr[2]
    rx_loc = right_fit_cr[0]*y_pixels**2+right_fit_cr[1]*y_pixels+right_fit_cr[2]
    
    lane_center_pixel = (right_fitx[0] + left_fitx[0])/2
    vehicle_offset = (vehicle_center - lane_center_pixel)*xm_per_pix
#    pct_difference = vehicle_offset/

    if vehicle_offset > 0:
        cv2.putText(newwarp,'Ego Vehicle is {} meters right of lane center'.format(vehicle_offset),(10,150), font, 1,(255,0,0),3,cv2.LINE_AA)
    if vehicle_offset < 0:   
        cv2.putText(newwarp,'Ego Vehicle is {} meters left of lane center'.format(vehicle_offset),(10,150), font, 1,(255,0,0),3,cv2.LINE_AA)
    if vehicle_offset == 0:
        cv2.putText(newwarp,'Ego Vehicle is directly on center!! Great job!',(10,150), font, 1,(255,0,0),3,cv2.LINE_AA)
        
# =============================================================================
#     This plots the lane line data for debugging vehicle center
# =============================================================================
#    plt.plot(lx_loc,y_pixels,'x')
#    plt.title('Left Lane Line Pixel Locations')
#    plt.show()
#  
#    plt.plot(rx_loc,y_pixels,'x')
#    plt.title('Right Lane Line Pixel Locations')
#    plt.show()  
#    
#    plt.plot(left_fitx,'x')
#    plt.plot(right_fitx,'o')
#    plt.title('Left Lane and Right Lane overlay, horizontal dir i "y" in image space')
#    plt.show()
#    
#    plt.figure(figsize = (15,15))
#    plt.imshow(newwarp)
#    plt.show()
#    
    # Combine the result with the original image
    #img = cv2.imread(img)
    img = cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)  
    
    #This is the final overlaid image with the texxto n it
#    plt.figure(figsize = (10,10))
#    plt.title('final result')
#    plt.imshow(result)
#    plt.show()
    
    return result, left_fitx, right_fitx, ploty

    

    
#--------------------- CAll functions and initiate camera cal and distortion corrrect-----------------------

#This section calls the camera calibration function

# Call the function to parse through the calibration image array and return
    #the base object point, corners and a grascale image for reference size


#***** TURN THIS ON LATER!!!!!! when you want to calibrate the camera
corners, imgpoints, objpoints, gray = calibrate_camera(image_dir, images)

##Generate the distortion coefficients and camera matrix, trans vector and rot vector
print('Generating distortion coefficients and camera matrix parameters')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)


#Undistort the images in the test_images folder
image_dir = "C:\\Users\\Mike\\Documents\\Self Driving Car nano-degree\\Term 1\\CarND-Advanced-Lane-Lines\\test_images\\"
images = os.listdir('test_images')
print('Selected image directory is: {} '.format(image_dir))
print('The images in the directory are: {}' .format(images))
distortion_corrected = distortion_correct(image_dir, images)
cv2.destroyAllWindows()


#--------------------- CAll functions to initiate a pipeline for image processing----------------------

image_dir = "C:\\Users\\Mike\\Documents\\Self Driving Car nano-degree\\Term 1\\CarND-Advanced-Lane-Lines\\test_images\\"
images = os.listdir('test_images')


print('Selected image directory is: {} '.format(image_dir))
print('The images in the directory are: {} \n' .format(images))
#print('The images in the directory are: {} \n' .format(images_new))

sobel_kernel = 9
#mag_thresh = [30,255]


#keep it
grad_threshold = [50,150]
sobel_mag = [0,255]



#distortion correct
if len(glob.glob('./test_images/*Distortion*.jpg')) == 0:
    print('there are no distortion corrected images in the directory, let us create them')
    distortion_corrected = distortion_correct(image_dir, images)

images = glob.glob('./test_images/*Distortion*.jpg')



def process_image(images):
#    for idx, fname in enumerate(images):
        
    img = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    orig_image = img
#        img = cv2.imread(fname)
#    plt.figure(figsize = (20,10))
#    plt.imshow(img)
    
    #pull in the absolute binary gradient data in X and Y
    gradx_binary = abs_sobel_image(img,'x',grad_threshold , sobel_kernel)
#    plt.figure(figsize = (20,10))
#    plt.title('Binary Gradient Thresholding in X direction')
#    plt.imshow(gradx_binary, cmap='gray')
#    plt.show()

    grady_binary = abs_sobel_image(img,'y',grad_threshold , sobel_kernel)
#    plt.figure(figsize = (20,10))
#    plt.title('Binary Gradient Thresholding in Y direction')
#    plt.imshow(grady_binary, cmap='gray')
#    plt.show()
    
    
    #Calculate the Sobel direction gradient binary threshold
    dir_binary = sobel_dir_thresh(img, sobel_kernel=15, thresh=(0.6, np.pi/2))
#    print(dir_binary.dtype)
#    plt.figure(figsize = (20,10))
#    plt.title('Binary Sobel (Absolute) Gradient Thresholding')
#    plt.imshow(dir_binary, cmap = 'gray')
    
#    mag_binary = sobel_mag_thresh(img, sobel_kernel, mag_thresh= (50, 150))
    mag_binary = sobel_mag_thresh(img, sobel_kernel, mag_thresh= (80, 150))
#    plt.figure(figsize = (20,10))
#    plt.title('Binary Gradient Magnitude Thresholding')
#    plt.imshow(mag_binary, cmap='gray')
#    mag_binary
    
    #Combine the gradient thresholds into a coherent image, there still may be gaps where color thresholding comes in
    combined_binary = np.zeros_like(dir_binary)
#    combined_binary[(gradx_binary == 1) | ((mag_binary == 1) | (dir_binary == 1))] = 1
    combined_binary[(gradx_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined_binary[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

#    plt.figure(figsize = (20,10))
#    plt.title('Combined Binary Gradient Thresholding (X,Mag,Dir)')
#    plt.imshow(combined_binary, cmap = 'gray')

    
    binary_color = color_binary(img, 'HLS', color_thresh = [80,255])
#    binary_color = color_binary(img, 'HLS', color_thresh = [80,180])
    
#    plt.figure(figsize = (20,10))
#    plt.title('Binary Color Thresholding in HLS')
#    plt.imshow(binary_color, cmap = 'gray')
#    plt.show()

    #Visualize the overall combined thresholding on the test images
    color_grad_combined = bin_color_gradient(combined_binary , binary_color) 
    
#    plt.figure(figsize = (20,10))
#    plt.title('Combined color and gradient mag thresholding')
#    plt.imshow(color_grad_combined, cmap = 'gray')
#    plt.show()
    
    img_size = img.shape
    offset = 100
    
    src = np.float32([(200, 720), (580, 480), (720, 480), (1050, 720)])
    dst = np.float32([(280, 720), (400, 190), (920, 190), (960, 720)])
    
    
    destination_points = np.float32([[offset, img_size[1]-offset], [img_size[0]-offset, img_size[1]-offset], 
                                     [img_size[0]-offset, offset], 
                                     [offset, offset]])
    
    source_points = np.float32(([450,780], [680, 1050], [680,250], [450, 500]))

    binary_warped, M, Minv = warp(color_grad_combined,src, dst)
    #warped_image_test = warp(img,source_points, destination_points)
    
#    plt.figure(figsize = (20,10)) 
#    plt.imshow(binary_warped, cmap='gray')
#    plt.show()
#    
#    
#    import numpy as np
#    plt.figure(figsize = (20,10))
#    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
#    plt.plot(histogram)
#    plt.show()
#    
    #Need the line data to be fed back out
    out, left_fitx, right_fitx, ploty = polyfit(binary_warped,img, Minv)
#    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
 


#
#trouble_images_from_movie = glob.glob('./test_images/out*.jpg')
#for idx, img in enumerate(trouble_images_from_movie):
#    test_image = cv2.imread(trouble_images_from_movie[idx])
#    test = process_image(test_image)
#    print('image {}'.format(idx) )
#    
    
    
#######--------------------------
##os.system("ffmpeg -i project_video.mp4 -vf fps=15/1 out_%03d.jpg'
Test_Video_dir = os.listdir("test_videos/")
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("test_videos/project_video.mp4")
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(video_output, audio=False)
  #-------------------------------------------  
    