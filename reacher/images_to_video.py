import cv2
import numpy as np
import glob
import os

def sortKey(s):
    name_file = os.path.basename(s)[4:-1]
    return int(name_file.strip('.jpeg'))


for i in range(1,9):
    img_array = []
    filename_all = []

    for filename in glob.glob('data/exp4_13-04-2019_19-51-33/episode%s00/*.jpg'%(i)):
        filename_all.append(filename)

    filename_all.sort(key=sortKey)
    for _, filename in enumerate(filename_all):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size =  (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('episode%s00_exp4.avi'%(i),cv2.VideoWriter_fourcc(*'MPEG'), 5, size)
    print('Steps episode %s00: '%(i), len(img_array))
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
