from math import sqrt, radians
from env_youbot import environment
import logz
from images_to_video import im_to_vid
import os

class youBot_controller(environment):
    def __init__(self):
        super().__init__(manipulator=False, base=True, obs_lowdim=True, rpa=1)
        # vehicle_ref = pr.get_dummy('youBot_vehicleReference')
        # vehicle_target = pr.get_dummy('youBot_vehicleTargetPosition')
        self.manipulator = False
        self.paramP = 25
        self.paramO = 5
        self.previousForwBackVel=0
        self.previousLeftRightVel=0
        self.previousRotVel=0
        self.accelF = 0.035
        self.maxV = 2
        self.maxVRot = 3
        self.camera = self.pr.get_vision_sensor('side_camera')
        #logz.configure_output_dir('demonstrations')

    # Individual controller for generating trajectories for the arm
    # for the base
    # Threshold to activate path search for arm once base in a radius of target
    # Add noise in trajectories and get different path to target and nearby
    # Store transitions, if reward function present: state, action, next_state, {reward}
    # Save in a txt file
    def get_action_to_target(self):
        done = False
        img_all = []
        a=0
        while not done:
            img = self.camera.capture_rgb()*256
            img_all.append(img)
            action_base, arm_status = self.base_actuation()
            self.step(action_base)
            if arm_status == True and a==0:
                a=1
                path = self.arm_actuation()
                if path is None:
                    print('NO PATH')
                    break

            if arm_status == True:
                done = path.step()
        return action_base, done, img_all

    def base_actuation(self):
        # This method is run at each simulation step
        pos_v = self.target.get_position(relative_to=self.base_ref)
        or_v = self.target.get_orientation(relative_to=self.base_ref)

        if sqrt(pos_v[0]**2 + pos_v[1]**2) < 0.35:
            return [0, 0, 0], True

        ForwBackVel = pos_v[1] * self.paramP
        LeftRightVel = pos_v[0] * self.paramP
        RotVel = or_v[2] * self.paramO
        v = sqrt(ForwBackVel*ForwBackVel+LeftRightVel*LeftRightVel)
        if v>self.maxV:
            ForwBackVel = ForwBackVel*self.maxV/v
            LeftRightVel = LeftRightVel*self.maxV/v

        if (abs(RotVel)>self.maxVRot):
            RotVel=self.maxVRot*RotVel/abs(RotVel)

        df = ForwBackVel- self.previousForwBackVel
        ds = LeftRightVel - self.previousLeftRightVel
        dr = RotVel - self.previousRotVel

        if (abs(df)>self.maxV*self.accelF):
            df=abs(df)*(self.maxV*self.accelF)/df

        if (abs(ds)>self.maxV*self.accelF):
            ds=abs(ds)*(self.maxV*self.accelF)/ds

        if (abs(dr)>self.maxVRot*self.accelF):
            dr=abs(dr)*(self.maxVRot*self.accelF)/dr


        ForwBackVel = self.previousForwBackVel+df
        LeftRightVel = self.previousLeftRightVel+ds
        RotVel = self.previousRotVel+dr

        self.previousForwBackVel = ForwBackVel
        self.previousLeftRightVel = LeftRightVel
        self.previousRotVel = RotVel

        return [ForwBackVel, LeftRightVel, RotVel], False

    def arm_actuation(self):
        path = self.arm.get_path(position=self.target.get_position(), orientation=[0, radians(180), 0])
        return path

    def generate_transitions(self, ep, obs_low=True, reward=False):
        self.reset()
        for _ in range(ep):
            while True:
                _, done = self.get_action_to_target()
                if done:
                    self.reset()
                    break

def main():
    if not(os.path.exists('demonstrations')):
        os.makedirs('demonstrations')
    imtovid = im_to_vid('demonstrations')
    controller = youBot_controller()
    for ep in range(10):
        controller.reset()
        _,_,img_all = controller.get_action_to_target()
        home = os.path.expanduser('~')
        os.chdir(os.path.join(home,'robotics_drl/reacher'))
        imtovid.from_list(img_all,ep)

    controller.terminate()

if __name__ == "__main__":
    main()
