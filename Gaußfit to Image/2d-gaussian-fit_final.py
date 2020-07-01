# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:59:27 2020

@author: grundch
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import pandas
import scipy.optimize as opt
from scipy import ndimage
import glob
import os

#%% Define consts and functs

SAVE_FIGURE = True
FSIZE = (11. / 2.54, 9. / 2.54)

CHOP_SIZE = 5 #multiples of sigma

#Pixelsize for beam width calculation
#   Point grey:3.75
#   IDS: 5.2
Pixelsize_x = 3.75
Pixelsize_y = Pixelsize_x

#%% Load data

DIR = os.getcwd()

filenames = (DIR + "/JL_LTAPO"+ ".png")


#%%
data = plt.imread(filenames)
# get image properties.
if (len(data.shape)<3):
    imgtype = 0
    h,w = np.shape(data)
else:
    imgtype = 1
    h,w,bpp = np.shape(data)

#%% plot data

plt.figure(1, figsize=FSIZE)

plt.title('Laser reflection image')
plt.imshow(data, cmap='gray', interpolation = 'bilinear')
#plt.xlabel('Pixel x / 1')
#plt.ylabel('Pixel y / 1')
#plt.colorbar()
#plt.grid()
plt.tight_layout()

#%%

def delete_defect_pixel(data, imgtype):
    if imgtype==1:
        pix = data[93, 107, 1]
        if pix == 1:
            data[93, 107, 1] = 0
    else:
        pix = data[93, 107]
        if pix == 1:
            data[93, 107] = 0
    return data

def chop_img(img):
    params = fitgaussian(img)
    (height, X_max, Y_max, sigma_x, sigma_y, offset) = params
    print(params)
    sigma_x = abs(sigma_x)
    sigma_y = abs(sigma_y)
    #width of gauss the function shall chop
    width_x = CHOP_SIZE*sigma_x
    width_y = CHOP_SIZE*sigma_y
    print('Area of Fit in x-Direction: {:4.2f}'.format(width_x))
    print('Area of Fit in y-Direction: {:4.2f}'.format(width_y))

    x_low = X_max-(width_x)
    x_high= X_max+(width_x)
    y_low = Y_max-(width_y)
    y_high= Y_max+(width_y)
    if x_low<0: x_low=0
    if x_high>img.shape[0]: x_high=img.shape[0]
    if y_low<0: y_low=0
    if y_high>img.shape[1]: y_high=img.shape[1]

    img = img[int(x_low):int(x_high), int(y_low):int(y_high)]

    return img

def rgb2gray(rgb):
    try:
        ret = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return ret
    except :
        print('Image seems to already be a gray scale image')
        return rgb


def gaussian(height, center_x, center_y, width_x, width_y, offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: offset + height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_y = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_x = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    offset = 0
    return height, x, y, width_x, width_y, offset

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = opt.leastsq(errorfunction, params)
    return p

def plot3D(data, title, SAVE_FIGURE, legend = False, legenddata=[]):
    Z = data
    X, Y = np.indices(data.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_xlim(0, len(X))
    ax.set_ylabel('Y')
    ax.set_ylim(0, len(Y))
    ax.set_zlabel('Z')
    ax.set_zlim(0, np.max(data))
    if legend:
        font = {
        'family': 'serif',
        'color' : 'black',
        'weight': 'normal',
        'size'  :  10
        }
        i=0.08
        for var in legenddata:
            ax.text2D(-0.08,i, r'$\frac{1}{e^2}$ in x: %s $\mu m$'%round(var,2), fontdict=font)
            i=i-0.02
    if SAVE_FIGURE: plt.savefig(title)

def plot3D_contour(data, title):
    Z = data
    X, Y = np.indices(data.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X, Y, Z, rstride=15, cstride=15, alpha=0.1)
    ax.set_title(title)
    cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset= np.max(Y), cmap=cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_xlim(0, len(X))
    ax.set_ylabel('Y')
    ax.set_ylim(0, len(Y))
    ax.set_zlabel('Z')
    ax.set_zlim(-0.4, np.max(data))
    plt.show()

def getContour(data, fit, SAVE_FIGURE):
    err = data-fit
    Y_max, X_max = np.where(fit == np.max(fit))
    row = data[:, Y_max]
    col = data[X_max, :].T
    row_fit = fit[:, Y_max]
    col_fit = fit[X_max, :].T
    row_err = err[:, Y_max]
    col_err = err[X_max, :].T
    plt.figure()
    plt.title('Slice through Max in x')
    #plt.imshow(data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    #plt.plot(X_max, Y_max, 'rx', markersize=70)
    plt.plot(row, 'k' ,label='slice')
    plt.plot(row_fit, 'r', label = 'fit')
    plt.plot(row_err, 'c', label = 'error')
    plt.legend(loc = 0)
    if SAVE_FIGURE: plt.savefig('slice_x')
    plt.figure()
    plt.title('Slice through Max in y')
    plt.plot(col, 'k' ,label='slice')
    plt.plot(col_fit, 'r', label = 'fit')
    plt.plot(col_err, 'c', label = 'error')
    plt.legend(loc = 0)
    if SAVE_FIGURE: plt.savefig('slice_y')


#%%preprocessing
data = delete_defect_pixel(data, imgtype)
#Convert image to grayscale
if imgtype==1: data = rgb2gray(data)
#Chop image
data = chop_img(data)

#%%Plot chopped, grayscaled image
plt.figure(2)
plt.imshow(data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

#%%Calculate and plot gaussian fit
params = fitgaussian(data)
fit = gaussian(*params)
fit_data = fit(*np.indices(data.shape))
plt.contour(fit_data, cmap=plt.cm.coolwarm)
ax = plt.gca()
(height, x, y, sigma_x, sigma_y, offset) = params

#%%Calculate the "real" sigma of beam with pixel size
real_x = sigma_x*Pixelsize_x
real_y = sigma_y*Pixelsize_y

#%%Calculate 1/(e^2) width -> 4*sigma
beamwidth_x = real_x *4
beamwidth_y = real_y *4
beam = [beamwidth_x, beamwidth_y]

font = {
        'family': 'serif',
        'color' : 'white',
        'weight': 'normal',
        'size'  :  10
}
plt.text(0,90, r'$\frac{1}{e^2}$ in x: %s $\mu m$'%round(beamwidth_y,2), fontdict=font)
plt.text(0,180, r'$\frac{1}{e^2}$ in y: %s $\mu m$'%round(beamwidth_x,2), fontdict=font)

if SAVE_FIGURE: plt.savefig('Beam_Fit')

#%%Error
div = data - fit_data
plt.figure(3)

#%% 3D Plots
plot3D(data, 'Beam profile', SAVE_FIGURE)
plot3D(fit_data, 'Gaussian', SAVE_FIGURE, True, beam)
plot3D(div, 'Error', SAVE_FIGURE)

#%% Contour Plots
getContour(data, fit_data, SAVE_FIGURE)