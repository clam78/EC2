# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:43:38 2023

@author: 19166
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:14:30 2023

@author: 19166
"""

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
# from PIL import Image, ImageDraw
import pdb
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.patches as mpatches

# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix


### Liquid form!
# Reading images from Google Drive 
# Read image response
import requests
# Full url on google drive
url = 'https://drive.google.com/file/d/159uQ2yoboe6ax-QEaRYiO4E3eQl3MCcY/view?usp=sharing'
# url = 'https://drive.google.com/file/d/159uQ2yoboe6ax-QEaRYiO4E3eQl3MCcY/view?usp=sharing'
# Extract file id to format url
file_id = url.split('/')[-2]
download_url = 'https://drive.google.com/uc?id=' + file_id
# Get response
response = requests.get(download_url)
img_array = np.frombuffer(response.content, np.uint8)
img_orig1 = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Decoded image

# show image. Needs color correction
img1plot = plt.imshow(img_orig1)

#convert to color. Image should be corrected
img_orig1Liq = cv2.cvtColor(img_orig1,cv2.COLOR_BGR2RGB)
img1LiquidPlot = plt.imshow(img_orig1Liq)
plt.title("RGB Img1 Plot")

# separate into r,g,b
imgLR = img_orig1Liq[:,:,0]; imgLG = img_orig1Liq[:,:,1]; imgLB = img_orig1Liq[:,:,2];

#generate subplots in same figure
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# ax1 = plt.imshow(imgR, cmap = "gray") 
# ax1.set_title('R')

# ax2 = plt.imshow(imgG) 
# ax2.set_title('G')

# ax3 = plt.imshow(imgB)                         #i attempted to use this to make subplots...
# ax3.set_title('B')


# img1LR = plt.im(imgR, cmap = "gray")    #plots R, G, and B band images separately. cmap can be messed with
# plt.title("Img1 R Band")
# # plt.show()                                  #ultimately we will want to make subplots, all in the same line
# img1LG = plt.im(imgG)
# plt.title("Img1 G Band")
# # plt.show()
# img1LB = plt.imshow(imgB)
# plt.title("Img1 B Band")
# # plt.show()

# code for displaying multiple images in one figure 
# create figure
fig = plt.figure(figsize=(10, 7))
# setting values to rows and column variables
rows = 1
columns = 3
  
# # reading images
# Image1 = cv2.imread(img1LR)
# Image2 = cv2.imread(img1LG)
# Image3 = cv2.imread(img1LB)
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(imgLR, cmap="gray")
plt.axis('off')
plt.title("Img1 R Band")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(imgLG, cmap="gray")
plt.axis('off')
plt.title("Img1 G Band")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(imgLB, cmap="gray")
plt.axis('off')
plt.title("Img1 B Band")

plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Split by RGB')
# ax1.plot()
# ax2.plot(x, -y)
# axs[0].set_xlim(2e8,7.5e8, 1e8)
# axs[0].set_ylim(30,90,10)
# axs[0].set_xlabel('C (J/(m^2 K))')
# axs[0].set_ylabel('tauC (yrs)')
# axs[0].grid(False)

# axs[0].set_title(scenario_title) #unindented these lines
    
# axs[1].set_xlim(2e8,7.5e8, 1e8)
# axs[1].set_ylim(7.0,11.0, 0.5)
# axs[0].set_xlabel('C (J/(m^2 K))')
# axs[1].set_ylabel('tauH (days)')
# axs[1].grid(False)
    
# axs[2].set_xlim(30,85, 10)
# axs[2].set_ylim(7.0,11.0, 0.5)
# axs[0].set_xlabel('tauC (yrs)')
# axs[2].set_ylabel('tauH (days)')
# axs[2].grid(False)

# # ANOTHER WAY. extract R, G, B bands to three new image variables
# # split image into different color channels
# b,g,r = cv2.split(img_orig1Liq)
# # define channel having all zeros
# zeros = np.zeros(b.shape, np.uint8)
# # merge zeros to make BGR image
# blueBGR = cv2.merge([b,zeros,zeros])
# greenBGR = cv2.merge([zeros,g,zeros])
# redBGR = cv2.merge([zeros,zeros,r])
# # display the three Blue, Green, and Red channels as BGR image
# cv2.imshow('Blue Channel', blueBGR)
# cv2.waitKey(0)
# cv2.imshow('Green Channel', greenBGR)
# cv2.waitKey(0)
# cv2.imshow('Red Channel', redBGR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# selecting pixels (and get their R,G,B) only from wax sample: add a mask
# 1. extract X,Y pixel coordinates to each pixel
Y3d, X3d, Z3d = np.meshgrid(np.arange(img_orig1Liq.shape[0]), np.arange(img_orig1Liq.shape[1]), np.arange(img_orig1Liq.shape[2]), indexing='ij')
X = X3d[:,:,0] # 2d slice
Y = Y3d[:,:,0] # 2d slice

# to plot both X and Y to make sure the values are what is expected
# I don't know if it's what we should expect: creates a colorful gradient of purple, blue, green, yellow
# plt.imshow(X)
# plt.imshow(Y)
   
# Plot the image
fig, ax = plt.subplots(1,1,figsize=(12, 12))
plt.title('Select the 4 edges of the sample', fontweight ="bold")
ax.imshow(img_orig1Liq)
# ax.imshow(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
# Hole locations
# 5x2 rows = 10 holes ; Ind 0 to 9
x1 = np.array( [ [(1160 + 350*i)]*5 for i in range(2)] ).flatten()
y1 = np.tile(1605 + 368*np.arange(5),2)
r1 = np.ones(len(x1))*100 #is this necessary?

# Equal aspect so circles look circular
ax.set_aspect('equal')
# Show the image
ax.imshow(img_orig1Liq)
# Now, loop through coord arrays, and create a circle at each x,y pair

for i in range(10):
    xy = (x1[i], y1[i])
    cv2.circle(img_orig1Liq, xy, 120, color = (255, 0, 0), thickness = 10)
cv2.imshow('Circle', img_orig1Liq)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# fig, ax = plt.subplots(1,1,figsize=(12, 12))
# # plt.title('Select the 4 edges of the sample', fontweight ="bold")
# ax.imshow(img_orig1Liq)



# ### Solid form!
# # Reading images from Google Drive 
# # Read image response
# import requests
# # Full url on google drive
# url = 'https://drive.google.com/file/d/1i4M6QGxhA2kVrvWhqfTl3u7Kb5gDTXUv/view?usp=sharing'
# # url = 'https://drive.google.com/file/d/1i4M6QGxhA2kVrvWhqfTl3u7Kb5gDTXUv/view?usp=sharing'
# # Extract file id to format url
# file_id = url.split('/')[-2]
# download_url = 'https://drive.google.com/uc?id=' + file_id
# # Get response
# response = requests.get(download_url)
# img_array = np.frombuffer(response.content, np.uint8)
# img_orig2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Decoded image

# # show image. Needs color correction
# img1plot = plt.imshow(img_orig2)

# #convert to color. Image should be corrected
# img_orig1Solid = cv2.cvtColor(img_orig2,cv2.COLOR_BGR2RGB)
# img1SolidPlot = plt.imshow(img_orig1Solid)

# # separate into r,g,b
# imgSR = img_orig1Solid[:,:,0]; imgSG = img_orig1Solid[:,:,1]; imgSB = img_orig1Solid[:,:,2];
# # img1SR = plt.imshow(imgR, cmap = "gray")    #plots R, G, and B band images separately. cmap can be messed with
# # plt.title("Img2 R Band")
# # plt.show()                                  #ultimately we will want to make subplots, all in the same line
# # img1SG = plt.imshow(imgG)
# # plt.title("Img2 G Band")
# # plt.show()
# # img1SB = plt.imshow(imgB)
# # plt.title("Img2 B Band")
# # plt.show()

# fig = plt.figure(figsize=(10, 7))
# # setting values to rows and column variables
# rows = 1
# columns = 3
  
# # Adds a subplot at the 1st position
# fig.add_subplot(rows, columns, 1)
  
# # showing image
# plt.imshow(imgSR, cmap="gray")
# plt.axis('off')
# plt.title("Img2 R Band")
  
# # Adds a subplot at the 2nd position
# fig.add_subplot(rows, columns, 2)
  
# # showing image
# plt.imshow(imgSG, cmap="gray")
# plt.axis('off')
# plt.title("Img2 G Band")
  
# # Adds a subplot at the 3rd position
# fig.add_subplot(rows, columns, 3)
  
# # showing image
# plt.imshow(imgSB, cmap="gray")
# plt.axis('off')
# plt.title("Img2 B Band")


