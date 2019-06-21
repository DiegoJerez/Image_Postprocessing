import os
import skimage
import numpy as np
import glob
import imageio as io
import numpy as np
import PIL
from PIL import Image
import imutils
import imageio as io
import cv2
import matplotlib.pyplot as plt


## OPEN IMAGE
image = np.array(Image.open('/home/diego/MPFI/Images/final_stitch_profile13.png'))
## MAX POOL
def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

max_pool = pooling(image, (3,3))

## CREATE THRESHOLDS FOR LABELS
# image_red = image[:,:,0] >
image_green = image[:,:,1] > 230
image_blue = image[:,:,2] < 40
image_green_max = max_pool[:,:,1] > 230
image_blue_max = max_pool[:,:,2] < 40
## COMBINE THRESHOLDS AS BOOLEANS
image_new = np.logical_and(image_green,image_blue)
image_new_max_pool = np.logical_and(image_green_max,image_blue_max)
## DISPLAY & SAVE
plt.imshow(image)
plt.show()
plt.imshow(image_new)
plt.show()
#plt.imsave('/home/diego/MPFI/Images/MPFI_algtest_threshold.png', image_new)
plt.imsave('/home/diego/MPFI/Images/final_stitch_labeled_profile13.png', image_new_max_pool )
