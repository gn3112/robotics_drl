import cv2
import numpy as np
import glob
import os

class im_to_vid(object):
    def __init__(self,exp_dir):
        home = os.path.expanduser('~')
        self.log_video_dir = os.path.join(home,'robotics_drl/',exp_dir,'episodes_video')
        if not(os.path.exists(self.log_video_dir)):
            os.makedirs(self.log_video_dir)

    def write_video(self,img_array,ep_num,size_img):
        size_img = (size_img[0],size_img[1])
        path = os.path.join(self.log_video_dir,'episode%s.avi'%(ep_num))
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MPEG'), 20, size_img)
        for i in range(len(img_array)):
            out.write(np.uint8(img_array[i])[...,::-1])

        out.release()

    def from_jpeg(self,exp_dir,episodes_number):
        def sortKey(self,s): #remove jpeg extension from string
            name_file = os.path.basename(s)[4:-1]
            return int(name_file.strip('.jpeg'))
        # episode_number is a list containing episodes to format to video
        # exp_file is the experiment log directory name
        for _, ep_num in enumerate(episodes_number):
            img_array = []
            filename_all = []

            for filename in glob.glob('data/' + exp_dir + '/episode%s/*.jpg'%(ep_num)): #get filename
                filename_all.append(filename)

            filename_all.sort(key=sortKey) #sort steps in order
            for _, filename in enumerate(filename_all):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size =  (width, height)
                img_array.append(img)

            print('Steps episode %s: '%(i), len(img_array))
            self.write_video(img_array,ep_num,size)

    def from_list(self,imgs,ep_num):
        height, width, layers = imgs[0].shape # Check data format
        size = [width, height]
        self.write_video(imgs,ep_num,size)
