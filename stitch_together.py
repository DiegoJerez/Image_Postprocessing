import os
import skimage
import numpy as np
import glob
import imageio as io
import numpy as np
import PIL
from PIL import Image

def stitch_row(n):
    file1 = np.array(Image.open('/home/diego/MPFI/gold_particles/pytorch-CycleGAN-and-pix2pix/results/gold_particle_model/test_latest/images/' + master[n]))
    file2 = np.array(Image.open('/home/diego/MPFI/gold_particles/pytorch-CycleGAN-and-pix2pix/results/gold_particle_model/test_latest/images/' + master[n+1]))
    full_row = np.concatenate((file1, file2), axis=1)
    for i in range(n + 2, n + 63):
        file_next = np.array(Image.open('/home/diego/MPFI/gold_particles/pytorch-CycleGAN-and-pix2pix/results/gold_particle_model/test_latest/images/' + master[i]))
        full_row = np.concatenate((full_row, file_next), axis = 1)
    return full_row

files = os.listdir('/home/diego/MPFI/gold_particles/pytorch-CycleGAN-and-pix2pix/results/gold_particle_model/test_latest/images')
list = []
for file in files:
    split_name = file.split('.')
    list.append(split_name[0])
list.sort(key = float)
master = []
for file in list:
    name = file + '.png_fake_B.png'
    master.append(name)

picture = stitch_row(0)
for n in range(63,2457,63):
    next_row = stitch_row(n)
    picture = np.concatenate((picture,next_row), axis=0)

io.imwrite('/home/diego/MPFI/Images/final_stitch_profile13.png', picture)
