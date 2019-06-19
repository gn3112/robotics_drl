from math import sqrt, abs
from env_youbot import environment
import logz
from images_to_video import im_to_vid

class youBot_controller(environemnt):
    def __init__(self):
        if not(os.path.exists('demonstrations')):
            os.makedirs('demonstrations')
        imtovid = im_to_vid('demonstrations')
        super().__init__(manipulator=False, base=True, obs_lowdim=True, rpa=1)
        # vehicle_ref = pr.get_dummy('youBot_vehicleReference')
        # vehicle_target = pr.get_dummy('youBot_vehicleTargetPosition')
        self.manipulator = False
        self.paramP = 20
        self.paramO = 10
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
            else:
                done = path.step()
        return action_base, done, img_all

    def base_actuation(self):
        # This method is run at each simulation step
        pos_v = self.target.get_position(relative_to=vehicle_ref)
        or_v = self.target.get_orientation(relative_to=vehicle_ref)

        if sqrt(pos_v[0]**2 + pos_v[1]**2) < 0.35:
            return [0, 0, 0], True

        ForwBackVel = pos_v[1] * paramP
        LeftRightVel = pos_v[0] * paramP
        RotVel = or_v[2] * param0
        v = math.sqrt(ForwBackVel*LeftRightVel+LeftRightVel*LeftRightVel)
        if v>self.maxV:
            ForwBackVel = ForwBackVel*self.maxV/v
            LeftRightVel = LeftRightVel*self.maxV/v

        if (math.abs(rotVel)>maxVRot):
            rotVel=maxVRot*rotVel/math.abs(rotVel)

        df = ForwBackVel- self.previousForwBackVel
        ds = LeftRightVel - self.previousLeftRightVel
        dr = RotVel - self.previousRotVel

        if (math.abs(df)>maxV*self.accelF):
            df=math.abs(df)*(maxV*self.accelF)/df

        if (math.abs(ds)>maxV*self.accelF):
            ds=math.abs(ds)*(maxV*self.accelF)/ds

        if (math.abs(dr)>maxVRot*self.accelF):
            dr=math.abs(dr)*(maxVRot*self.accelF)/dr


        forwBackVel = self.previousForwBackVel+df
        leftRightVel = self.previousLeftRightVel+ds
        rotVel = self.previousRotVel+dr

        self.previousForwBackVel = forwBackVel
        self.previousLeftRightVel = leftRightVel
        self.previousRotVel = rotVel

        return [forwBackVel, leftRightVel, rotVel], False

    def arm_actuation(self):
        path = self.arm.get_path(position=self.target.get_position(), orientation=[0, math.radians(180), 0])
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
    for ep in range(10):
        controller = youBot_controller()
        controller.get_action_to_target()
        home = os.path.expanduser('~')
        os.chdir(join(home,'robotics_drl/reacher'))
        imtovid.from_list(img_all,ep)

if __name__ == "__main__":
    main()
