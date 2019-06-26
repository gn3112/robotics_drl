from math import sqrt, radians
from env_youbot import environment
import logz
from images_to_video import im_to_vid
import os
from probabilistic_road_map import PRM_planning
import numpy as np

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
        self.intermediate_target = self.pr.get_dummy('intermediate_target')
        self.img_all = []
        #logz.configure_output_dir('demonstrations')

    # Individual controller for generating trajectories for the arm
    # for the base
    # Threshold to activate path search for arm once base in a radius of target
    # Add noise in trajectories and get different path to target and nearby
    # Store transitions, if reward function present: state, action, next_state, {reward}
    # Save in a txt file
    
    def intermediate_points_base(self):
        start = self.youBot.get_position()[:2]
        goal = self.goal[0,:]
        pts = []
        
        ox = []
        oy = []
        wl = 2
        for i in range(-wl,wl):
            ox.append(i)
            oy.append(-wl)
        for i in range(-wl,wl):
            ox.append(-wl)
            oy.append(i)
        for i in range(-wl,wl+1):
            ox.append(i)
            oy.append(wl)
        for i in range(-wl,wl+1):
            ox.append(wl)
            oy.append(i) 
        print("start",start)
        print('goal',goal)
        rx, ry = PRM_planning(start[0],start[1],goal[0],goal[1],ox,oy, 0.05) # robot size 0.05
        rxy = np.flip(np.column_stack((rx,ry)),axis=0)
        print(rxy)
        num_nodes = np.shape(rxy)[0]
        
        if num_nodes < 4:
            return rxy.tolist()

        for i in range(3,num_nodes,3):
            if i == num_nodes//3 * 3: 
                pts.append(rxy[-1,:])
                continue

            pts.append(rxy[i,:])

        return pts

    def is_base_reached(self):
        pos_v = self.intermediate_target.get_position(relative_to=self.base_ref)
        if sqrt(pos_v[0]**2 + pos_v[1]**2) < 0.005:
            return True

    def generate_trajectories(self):
        # Reaching with base and then with manipulator
        self.img_all = []
        base_status = self.generate_base_trajectories()
        if base_status == False: 
            done = self.generate_arm_trajectories()
            return done        
        else:
            return False

    def follow_path(self,path):
        done = False
        while not done:
            self.store_transitions()
            done = path.step()
        return done
    
    def generate_base_trajectories(self):
        steps = 0
        base_status = True
        inter_pts = self.intermediate_points_base()
        i_pts = 0 
        self.img_all = []
        while base_status != False:
            self.intermediate_target.set_position([inter_pts[i_pts][0],inter_pts[i_pts][0],0.1])
            
            if self.is_base_reached():
                i_pts += 1

            steps += 1
            self.store_transitions()
            action_base, base_status = self.get_base_actuation()
            self.step(action_base) 
            if steps > 300:
                break
        return base_status

    def generate_arm_trajectories(self):
        path = self.get_arm_path()
        
        if path is None:
            return False
        
        done = self.follow_path(path) 
        
        return done

    def store_transitions(self):
        img = self.camera.capture_rgb()*256
        self.img_all.append(img)
    
    def get_arm_path(self):
        path = self.arm.get_path(position=self.target.get_position(), orientation=[0, radians(180), 0])
        return path 
   
    def get_base_actuation(self):
        # This method is ran at each simulation step
        pos_v = self.intermediate_target.get_position(relative_to=self.base_ref)
        or_v = self.intermediate_target.get_orientation(relative_to=self.base_ref)
        
        pos_youBot = self.youBot.get_position()
        if sqrt((self.goal[0][0]-pos_youBot[0])**2 + (self.goal[0][1]-pos_youBot[1])**2) < 0.01:
            return [0, 0, 0], False

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

        return [ForwBackVel, LeftRightVel, RotVel], True

    def generate_transitions(self, ep, obs_low=True, reward=False):
        self.reset()
        for _ in range(ep):
            while True:
                _, done = self.get_action_to_target()
                if done:
                    self.reset()
                    break

def main():
    
    if not(os.path.exists('data/demonstrations')):
        os.makedirs('data/demonstrations')
    imtovid = im_to_vid('data/demonstrations')
    
    controller = youBot_controller()
    
    for ep in range(5):
        controller.reset()
        target_pos = controller.target.get_position()
        youBot_pos = controller.youBot.get_position()
        controller.goal = np.array([target_pos[:2],[0,0]]) 
        done = controller.generate_base_trajectories()
        print(done)
        
        img_all = controller.img_all
        
        home = os.path.expanduser('~')
        os.chdir(os.path.join(home,'robotics_drl/reacher'))
        
        if len(img_all) > 1: imtovid.from_list(img_all,ep)
        
        controller.reset()

    controller.terminate()

if __name__ == "__main__":
    main()
