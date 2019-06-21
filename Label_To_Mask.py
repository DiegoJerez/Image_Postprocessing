import scipy
from scipy import ndimage
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import labeled_comprehension as extract_feature
import cv2
from PIL import Image
import imutils
import imageio

Image.MAX_IMAGE_PIXELS = None

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


def load_image_path(img_path):
    img = imageio.imread(img_path) # gray-scale image
    return img


def red_particles(img):
    img_red = img[:,:,0] > 170
    img_green = img[:,:,1] < 70
    img_blue = img[:,:,2] < 80
    img_new = np.logical_and(img_red, img_green)
    img_new = np.logical_and(img_new, img_blue)
    img = img_new.astype(np.uint8)
    img[img > 0] = 255
    return img

def green_particles(img):
    img_green = img[:,:,1]>200
    img_red = img[:,:,0]<147
    img_blue = img[:,:,2] < 30
    img_new = np.logical_and(img_green, img_red)
    img_new = np.logical_and(img_new, img_blue)
    img = img_new.astype(np.uint8)
    img[img>0]=255
    return img

def disconnect_and_pad(image):
    for rows in image:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        for i in range(128):
            a = b
            b = c
            c = d
            d = e
            e = rows[i]
            if (a == b == c == d == e == 255):
                rows[i] = 0
                e = 0

    for rows in image:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        for i in range(128):
            a = b
            b = c
            c = d
            d = e
            e = rows[i]
            if (a == b == d == e == 0 and c == 255):
                rows[i-1] = 255
                d = 255

    transposed = np.transpose(image)
    for rows in transposed:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        for i in range(128):
            a = b
            b = c
            c = d
            d = e
            e = rows[i]
            if (a == b == c == d == e == 255):
                rows[i] = 0
                e = 0

    for rows in transposed:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        for i in range(128):
            a = b
            b = c
            c = d
            d = e
            e = rows[i]
            if (a == b == d == e == 0 and c == 255):
                rows[i-1] = 255
                d = 255
    image = np.transpose(transposed)
    return image

def disconnect_small(image):
    width = image.shape[0]
    long = image.shape[1]
    for rows in image:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        for i in range(long):
            a = b
            b = c
            c = d
            d = e
            e = f
            f = g
            g = h
            h = rows[i]
            if (a == b == c == d == e == f == g == h == 255):
                rows[i] = 0
                h = 0

    transposed = np.transpose(image)
    for rows in transposed:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        for i in range(width):
            a = b
            b = c
            c = d
            d = e
            e = f
            f = g
            g = h
            h = rows[i]
            if (a == b == c == d == e == f == g == h == 255):
                rows[i] = 0
                h = 0

    image = np.transpose(transposed)
    return image

def disconnect_large(image):
    width = image.shape[0]
    long = image.shape[1]
    for rows in image:
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = 0
        p8 = 0
        p9 = 0
        p10 = 0
        p11 = 0
        p12 = 0
        p13 = 0
        p14 = 0
        p15 = 0
        p16 = 0
        for i in range(long):
            p1 = p2
            p2 = p3
            p3 = p4
            p4 = p5
            p5 = p6
            p6 = p7
            p7 = p8
            p8 = p9
            p9 = p10
            p10 = p11
            p11 = p12
            p12 = p13
            p13 = p14
            p14 = p15
            p15 = p16
            p16 = rows[i]
            if (p1 == p2 == p3 == p4 == p5 == p6 == p7 == p8 == p9 == p10 == p11 == p12 == p13 == p14 == p15 == p16 == 255):
                rows[i] = 0
                p16 = 0

    transposed = np.transpose(image)
    for rows in transposed:
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = 0
        p8 = 0
        p9 = 0
        p10 = 0
        p11 = 0
        p12 = 0
        p13 = 0
        p14 = 0
        p15 = 0
        p16 = 0
        for i in range(width):
            p1 = p2
            p2 = p3
            p3 = p4
            p4 = p5
            p5 = p6
            p6 = p7
            p7 = p8
            p8 = p9
            p9 = p10
            p10 = p11
            p11 = p12
            p12 = p13
            p13 = p14
            p14 = p15
            p15 = p16
            p16 = rows[i]
            if (p1 == p2 == p3 == p4 == p5 == p6 == p7 == p8 == p9 == p10 == p11 == p12 == p13 == p14 == p15 == p16 == 255):
                rows[i] = 0
                p16 = 0

    image = np.transpose(transposed)
    return image

def get_coordinates(image):
    cnts = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            M["m00"] = 1
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        if (cX != 0 or cY != 0):
            print(cX,cY)
        # cv2.circle(newlabeled, (cX, cY), 2,(255,255,255), -1)
        # cv2.putText(image, "center", (cX - 4, cY - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        # cv2.imshow(image)
        # cv2.waitKey(0)


##### RUN
img_path = '/home/diego/MPFI/Images/Original_ColorOverlays/labeled.png'
img = load_image_path(img_path)
#small_gold_particles = red_particles(img)
large_gold_particles = green_particles(img)
blue_layer = np.zeros((7461,5655), dtype = np.uint8)

labeled_final = np.stack((blue_layer, large_gold_particles, blue_layer), axis = -1)
plt.imshow(labeled_final)
#plt.imshow(img)
plt.show()
imageio.imwrite('/home/diego/MPFI/Images/CycleGAN_onlyGreen/Labels/onlygreen_mask_2.png', labeled_final)
# print('These are the x,y for small gold particles')
# get_coordinates(small_gold_particles)
# print('These are the x,y for large gold particles')
# get_coordinates(large_gold_particles)


# structure = [[1,1,1],
#             [1,1,1],
#             [1,1,1]]
# labeled_small, count_small = ndimage.label(small_gold_particles, structure = structure)
# print("Number of small gold objects is %d" % count_small)
# labeled_large, count_large = ndimage.label(large_gold_particles, structure = structure)
# print("Number of large gold objects is %d" % count_large)
