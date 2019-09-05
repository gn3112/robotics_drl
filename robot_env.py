from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from math import pi, sqrt
import random
from os.path import dirname, join, abspath
import os

class robotEnv(object):
    def __init__(self, scene_name, reward_dense, boundary):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), scene_name)
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.pr.set_simulation_timestep(0.05)

        if scene_name != 'robot_scene.ttt': #youbot_navig2
            home_dir = os.path.expanduser('~')
            os.chdir(join(home_dir,'robotics_drl'))
            self.pr.import_model('robot.ttm')

        # Vision sensor handles
        self.camera_top = VisionSensor('Vision_sensor')
        self.camera_top.set_render_mode(RenderMode.OPENGL3)
        self.camera_top.set_resolution([256,256])

        self.camera_arm = VisionSensor('Vision_sensor1')
        self.camera_arm.set_render_mode(RenderMode.OPENGL3)
        self.camera_arm.set_resolution([128,128])

        self.reward_dense = reward_dense
        self.reward_termination = 1 if self.reward_dense else 0
        self.boundary = boundary

        # Robot links
        robot_links = []
        links_size = [3,5,5,1]
        for i in range(len(links_size)):
            robot_links.append([Shape('link%s_%s'%(i,j)) for j in range(links_size[i])])

        links_color = [[0,0,1],[0,1,0],[1,0,0]]
        color_i = 0
        for j in robot_links:
            if color_i > 2:
                color_i = 0
            for i in j:
                i.remove_texture()
                i.set_color(links_color[color_i])
            color_i += 1

    def render(self,view='top'):
        if view == 'top':
            img = self.camera_top.capture_rgb()*256 # (dim,dim,3)
        elif view == 'arm':
            img = self.camera_arm.capture_rgb()
        return img

    def terminate(self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()

    def sample_action(self):
        return [(2 * random.random() - 1) for _ in range(self.action_space)]

    def rand_bound(self):
        x = random.uniform(-self.boundary,self.boundary)
        y = random.uniform(-self.boundary,self.boundary)
        orientation = random.random() * 2 * pi
        return x, y, orientation

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        _, obs = self.get_observation()
        return obs.shape

    def step_limit(self):
        return 250
