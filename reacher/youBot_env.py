from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from math import pi, sqrt
import random
from os.path import dirname, join, abspath

class youBotEnv(object):
    def __init__(self, reward_dense=True, boundary=1, scene_name: str = 'youbot_navig.ttt'):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), scene_name)
        self.pr.launch(SCENE_FILE,headless=True)
        self.pr.start()

        if scene_name != 'youbot_navig.ttt':
            self.pr.import_model('youbot.ttm')

        # Vision sensor handles
        self.camera_top = VisionSensor('Vision_sensor')
        self.camera_arm = VisionSensor('Vision_sensor0')

        self.reward_dense = reward_dense
        self.reward_termination = 1 if self.reward_dense else 0
        self.boundary = boundary

    def render(self,view='top'):
        if view == 'top':
            img = self.camera_top.capture_rgb() # (dim,dim,3)
        elif view == 'arm':
            img = self.camera_arm.capture_rgb()
        return img*256

    def terminate(self):
        self.pr.start()  # Stop the simulation
        self.pr.shutdown()

    def sample_action(self):
        return [(2 * random.random() - 1) for _ in range(self.action_space)]

    def rand_bound(self):
        xy_min = 0
        xy_max = self.boundary - 0.2
        x = random.uniform(xy_min,xy_max)

        y_max = sqrt(xy_max**2-x**2)
        y_min = 0
        y = random.uniform(y_min,y_max)

        return x, y

    def action_type(self):
        return 'continuous'

    def observation_space(self):
        _, obs = self.get_observation()
        return obs.shape

    def step_limit(self):
        return 250
