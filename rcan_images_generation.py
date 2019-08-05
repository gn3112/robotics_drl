from env_youbot import environment
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import TextureMappingMode
import numpy as np
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from random import randrange, random
from skimage import io
from math import sqrt

from torch.utils.data import Dataset, DataLoader

class RCAN_Dataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

    def __len__(self):
        self._set_dir()
        return len([name for name in os.listdir('.') if os.path.isfile(name)])//3

    def __getitem__(self,idx):
        self._set_dir()

        x_image = io.imread('%s_input.png'%(idx))

        y_images = []
        for i in range(2):
            y_images.append(io.imread('%s_output%s.png'%(idx,i)))

        sample = {'input': x_image, 'output':y_images}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _set_dir(self):
        home = os.path.expanduser('~')
        dir = os.path.join(home,self.dir)
        os.chdir(dir)

class RCAN_generate_data():
    def __init__(self, n_samples, scene_pool):
        # 250K images
        self.env = environment(scene_name='scene1.ttt',manipulator=False, base=True, obs_lowdim=True)

        camera = VisionSensor('Vision_sensor1')

        # handles for textures change
        walls = [Shape('wall%s'%(i+1)) for i in range(4)]
        target = Shape('target')
        floor = Shape('Floor')

        self.env.reset()

    def __call__(self):
        for i in range(n_samples):
            if i % 100 == 0 and i != 0:
                self._generate_new_scene()
        # 1_input
        # 1_output0
        # 1_output1
        # Get initial textures
        # rgb_img_can = camera.capture_rgb()
        # rgb_d_img = camera.capture_depth()
        #
        # # texture_id1 = get_texture(randrange(6))
        # # texture_id2 = get_texture(randrange(6))
        # # texture_object, texture_id = env.pr.create_texture(filename='/home/georges/robotics_drl/reacher/temp.png',interpolate=True, decal_mode=True,
        # # repeat_along_u=True, repeat_along_v=True)
        # i=0
        # target.set_texture(texture = get_texture(4), mapping_mode = TextureMappingMode.SPHERE, decal_mode=True,repeat_along_u=True, repeat_along_v=True)
        #
        # for j in walls:
        #     j.set_texture(texture = get_texture(i), mapping_mode = TextureMappingMode.PLANE, decal_mode=False,repeat_along_u=True, repeat_along_v=True, uv_scaling=[1., 1.])
        #     i+= 1
        #
        #
        # floor.set_texture(texture = get_texture(6), mapping_mode = TextureMappingMode.PLANE,decal_mode=False, repeat_along_u=True, repeat_along_v=True)
        #
        # for _ in range(1):
        #     env.pr.step()
        #
        # # target.remove_texture()
        # # [j.remove_texture() for j in walls]S
        # # floor.remove_texture()
        # # target.set_texture(texture = get_texture(4), mapping_mode = TextureMappingMode.PLANE,decal_mode=False)
        #
        # # Generate rgb canonical, rgb randomised, depth
        #
        # rgb_img_rand = camera.capture_rgb()
        #
        # plt.imshow(rgb_d_img,cmap='gray')
        # plt.imshow(rgb_img_rand)
        # plt.show()
        #
        # # Save images
        # Img_1 = Image.fromarray(np.uint8(rgb_img_can*255),'RGB')
        # Img_2 = Image.fromarray(np.uint8(rgb_d_img*255))
        # Img_3 = Image.fromarray(np.uint8(rgb_img_rand*255),'RGB')
        #
        # home_dir = os.path.expanduser('~')
        # os.chdir(os.path.join(home_dir,'robotics_drl/reacher'))
        # Img_1.save('rgb_canonical.png')
        # Img_2.save('rgb_d.png')
        # Img_3.save('rgb_rand.png')

        self.env.reset()

        self.env.terminate()

    def _generate_new_scene(self):
        try:
            env.terminate()
        except:
            pass

        env = environment(scene='scene1.ttt',manipulator=False, base=True, obs_lowdim=True)
        walls = [Shape('wall%s'%(i+1)) for i in range(4)]
        target = Shape('target')
        floor = Shape('Floor')

        wall_dim = walls[0].get_bounding_box()
        area_scene = wall_dim[-1] ** 2
        n_objects = round(area_scene * 0.5)

        objects = []
        objects_eval = []
        for k in range(n_objects):
            obj_no = random.random(999)
            objects.append(Shape.import_mesh('%s/%s_coll.obj'%(obj_no), scaling_factor=0.1))

            # Register object pos
            # Check object is far by a distance from all objects
            while True:
                pos_objects.append([random.randrange(-1,1) for _ in range(2)])
                objects[k].set_position([pos_objects[k][0],pos_objects[k][1],0.5])
                self.env.pr.step()
                if objects[k].check_collision() != True:
                    break

                # for j in range(len(pos_objects)-1):
                #     dist = sqrt((pos_objects[-1][0]-pos_objects[j][0])**2 + (pos_objects[-1][1]-pos_objects[j][0])**2)

                    # if dist > 0.4:
                    #     pass
                    #     # or get angle
                    # else:
                    #     del objects_eval[j]
                    #
                    # if len(objects_eval) == 0:
                    #     break

            objects_eval = objects





# Load and apply texture
def get_texture(n_texture):
    all = ['zigzagged_0124','blotchy_0015','chequered_0179','blotchy_0022','banded_0153','banded_0150','chequered_0175']
    texture_object, texture_id = env.pr.create_texture(filename='/home/georges/robotics_drl/reacher/textures/obj_textures/%s.png'%(all[n_texture]))
    return texture_id

def main():
    gen = RCAN_generate_data(1,1)
    gen()

if __name__ == "__main__":
    main()
