from math import sqrt, abs
from env_youbot import environment

class youBot_controller(environemnt):
    def __init__(self):
        super().__init__()
        vehicle_ref = pr.get_dummy('youBot_vehicleReference')
        vehicle_target = pr.get_dummy('youBot_vehicleTargetPosition')
        self.paramP = 20
        self.paramO = 10
        self.previousForwBackVel=0
        self.previousLeftRightVel=0
        self.previousRotVel=0
        self.accelF = 0.035
        self.maxV = 2
        self.maxVRot = 3

    def base_actuation(self):
        pos_v = vehicle_target.get_position(relative_to=vehicle_ref)
        or_v = vehicle_target.get_orientation(relative_to=vehicle_ref)

        ForwBackVel = pos_v[1] * paramP
        LeftRightVel = pos_v[0] * paramP
        RotVel = or_v[2] * param0
        v = math.sqrt(ForwBackVel*LeftRightVel+LeftRightVel*LeftRightVel)
        if v>self.maxV:
            ForwBackVel = ForwBackVel*self.maxV/v
            LeftRightVel = LeftRightVel*self.maxV/v

        if (math.abs(rotVel)>maxVRot):
            rotVel=maxVRot*rotVel/math.abs(rotVel)

        df=ForwBackVel-previousForwBackVel
        ds=LeftRightVel-previousLeftRightVel
        dr=RotVel-previousRotVel

        if (math.abs(df)>maxV*self.accelF):
            df=math.abs(df)*(maxV*self.accelF)/df

        if (math.abs(ds)>maxV*self.accelF):
            ds=math.abs(ds)*(maxV*self.accelF)/ds

        if (math.abs(dr)>maxVRot*self.accelF):
            dr=math.abs(dr)*(maxVRot*self.accelF)/dr


        forwBackVel=previousForwBackVel+df
        leftRightVel=previousLeftRightVel+ds
        rotVel=previousRotVel+dr

        set_wheel_vel()

        previousForwBackVel=forwBackVel
        previousLeftRightVel=leftRightVel
        previousRotVel=rotVel

    def set_vehicle_target_positon(self):
        return None
