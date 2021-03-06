# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:49:53 2020

@author: CKvG
"""


import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

#Define if image should have noise
noisy = True
noise_amp = 0.005

def twoD_Gaussian(data, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = data
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

# Create x and y indices/ Size of picture
x = np.linspace(0, 1280, 1281)
y = np.linspace(0, 1024, 1025)
x, y = np.meshgrid(x, y)
data = (x,y)

#create data
image_data = twoD_Gaussian(data, 0.38, 518, 621, 50, 50, 0, 0.12)
if noisy: image_data = image_data + noise_amp*np.random.normal(size=image_data.shape)

# plot twoD_Gaussian data generated above
fig = plt.figure(frameon=False, figsize=(12.80, 10.24), dpi=100)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
image = image_data.reshape(1025,1281)
ax.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

#fig.savefig('Gaussian_Picture.png', cmap=plt.get_cmap('gray'))
