# Need a def to generate structural space definition (number of walls entities, heights, ceiling high depending on areas, heights)
from random import random, randrange, uniform, choice
import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.light import Light
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from os.path import dirname, join, abspath, expanduser
import os
import math
from youBot_all import youBotAll
from pyrep.const import TextureMappingMode
from PIL import Image
import json
import time

def visualise(walls):
    a = np.empty((32,32),dtype=float)
    a[3:28,1] = 1
    plt.imshow(a)
    plt.show()
    return None

def create_structure(n_iteration):
    # Steps:
    # Constants: min wall size, min wall height, min max area m^2, number of adding iterations
    # Create an inital rectangle (determine each wall position and or_w)
    # Get x and y edges to attach next entity (Rectangle as well)
    # Determine their wall position
    # Append to wall lists
    # Determine new x and y possible edges

    # Declare initial wall dimensions
    min_wall_l = 1
    max_wall_l = 3
    min_wall_h = 2
    max_wall_h = 3

    walls = {}
    closed_walls = {}

    lengths = []
    for _ in range(2):
        lengths.append(random() * (max_wall_l - min_wall_l) + min_wall_l)

    lengths = lengths.sort()

    h = random() * (max_wall_h - min_wall_h) + min_wall_h

    k = 0
    sign = [-1,1]
    orient = [0,90]

    coord = [[lengths[1],0],[-lengths[1],0],[lengths[0],0],[-lengths[0],0]]
    for i in range(4):
        if i>1: k=k+1
        # Short first and 0
        walls.update({'1_%s'%(i+2): [lenghts[k], h, coord[i], orient[k]]})

    # Register all boundary walls
    closed_walls = walls


    for it in range(n_iteration):
        n_closed_sides = 4

        along_side = randrange(1, n_closed_sides+1)

        side_to_built = walls['1_%s'%(along_side)] # Replace by available edges

        n_closed_sides = n_closed_sides + 3

        min_wall_l = side_to_built[0]/3
        max_wall_l = side_to_built[0]

        lengths = []
        for _ in range(2):
            lengths.append(random() * (max_wall_l - min_wall_l) + min_wall_l)

        lengths = lengths.sort()

        long_free = randrange(2)
        if long_free == 0:
            colinear = lengths[-1]
        else:
            colinear = lenghts[0]

        center_along = random() * (new_max_wall_l - colinear)
        #if side_to_built[-1] == 0 else
        center_along = center_along - new_max_wall_l/2

        h_equal = randrange(2)
        if h_equal == 0:
            h = random() * (max_wall_h - min_wall_h) + min_wall_h

        coord = [[lengths[1],0],[-lengths[1],0],[lengths[0],0],[-lengths[0],0]]
        # Shift coord from original
        shift_coord = [side_to_built[i][0] + (center_along if side_to_built[i][-1] == 0 else 0), side_to_built[i][1] + (center_along if side_to_built[i][-1] == 90 else 0)]
        for i in range(4):
            for j in range(2):
                coord[i][j] = coord[i][j] + shift_coord[j]

        k = 0
        sign = [-1,1]
        orient = [0,90]
        for i in range(4):
            if i>1: k=k+1
            # Short first and 0
            walls.update({'%s_%s'%(it+2,i+2): [lenghts[k], h, coord[i], orient[k]]})

        # Two new segments 1_4 1_5


        closed_walls.pop('1_%s'%(along_side))
        closed_wall.update()

    return closed_walls

def create_random_scene():
    # Constants
    width_w = 2
    height_w = 2.3
    dist_walls_front = 2
    dist_walls_side = 3

    walls = [[],[]]

    # Add wall
    for w_n in range(5):
        width_w = uniform(3,3)
        or_w = randrange(2)

        if or_w:
            walls[or_w].append(Shape.create(PrimitiveShape.CUBOID,[0.01,width_w,height_w],orientation=[0,0,math.radians(90)], static=True))
        else:
            walls[or_w].append(Shape.create(PrimitiveShape.CUBOID,[0.01,width_w,height_w],orientation=[0,0,0], static=True))

        x_pos = uniform(-2.5,2.5)
        y_pos = uniform(-2.5,2.5)
        walls[or_w][-1].set_position([x_pos,y_pos,height_w/2])

        if w_n == 0:
            continue

        while True:
            x_pos = uniform(-2.5,2.5)
            y_pos = uniform(-2.5,2.5)

            walls[or_w][-1].set_position([x_pos,y_pos,height_w/2])

            c = True
            for k in range(len(walls[or_w])-1):
                pos_rel = walls[or_w][-1].get_position(relative_to = walls[or_w][k])
                dist_front = abs(pos_rel[or_w])
                dist_side = abs(pos_rel[int(not or_w)])
                if dist_front < dist_walls_front and dist_side < dist_walls_side:
                    c = False
                    break

            if c == True:
                break

        walls[or_w][-1].set_collidable(1)

    return walls

def add_procedural_object():
    n = str(randrange(0,999))
    if len(n) < 3:
        for _ in range(3 - len(n)):
            n = '0' + n

    home_dir = expanduser('~')
    dir = join(home_dir,'robotics_drl/random_urdfs/%s/%s_coll.obj'%(n,n))
    a = Shape.import_mesh(filename=dir,scaling_factor=0.05)
    # Create convex collidable object
    b = a.get_convex_decomposition(use_vhacd=True, vhacd_res=100)
    a.set_parent(b,keep_in_place=True)

    b.set_collidable(0)
    b.set_measurable(0)
    b.set_detectable(0)
    b.set_renderable(0)

    b.set_respondable(1)
    b.set_dynamic(1)

    a.set_collidable(1)
    a.set_measurable(1)
    a.set_detectable(1)
    a.set_renderable(1)

    b.reset_dynamic_object()
    pos_2d = [uniform(-5, 5) for _ in range(2)]
    b.set_position(pos_2d + [3])

    return a, b

def add_object():
    while True:
        obj_dir = choice([name for name in os.listdir('/home/georges/Downloads/models') if name[-3:] == 'obj'])
        obj_mesh_file = join('/home/georges/Downloads/models', obj_dir)
        a = Shape.import_mesh(filename=obj_mesh_file,scaling_factor=0.01)
        box_size = a.get_bounding_box()
        height = box_size[1]
        width = box_size[-1]
        length = box_size[3]
        if height > 1 or width > 1 or length > 1 or height < 0.05 or width < 0.05 or length < 0.05:
            a.remove()
        else:
            break

    # Create convex collidable object

    b = a.get_convex_decomposition(use_vhacd=True, vhacd_res=100)
    box_size = a.get_bounding_box()
    # b = Shape.create(PrimitiveShape.CUBOID,[box_size[-1],box_size[3],box_size[1]],orientation=[0,0,0], static=True)

    a.set_parent(b,keep_in_place=True)
    a.set_position([0,0,0],relative_to=b)

    b.set_collidable(0)
    b.set_measurable(0)
    b.set_detectable(0)
    b.set_renderable(0)

    b.set_respondable(1)
    b.set_dynamic(1)

    a.set_collidable(1)
    a.set_measurable(1)
    a.set_detectable(1)
    a.set_renderable(1)

    b.reset_dynamic_object()

    iter = 0
    while a.check_collision():
        iter += 1
        pos_2d = [uniform(-5, 5) for _ in range(2)]
        b.set_position(pos_2d + [3])
        if iter > 12:
            break

    a_all = a.ungroup()
    for idx, j in enumerate(a_all):
        a_all[idx] = Shape(j)

    return a_all, b

def load_scene():
    home_dir = expanduser('~')
    dir = join(home_dir,'scenenet_scenes/1Bathroom/5_labels.obj')
    a = Shape.import_mesh(filename=dir)

def apply_domain_rand(floor, camera,
            lights, walls, ceiling, objects, env):
    # Apply domain rand to: lightning, texture, position camera, field of view
    texture_objects = []

    # texture_id, texture_object = _get_texture(env)
    # texture_objects.append(texture_object)
    # target.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.CUBE)

    texture_id, texture_object = _get_texture(env)
    texture_objects.append(texture_object)
    floor.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=True, repeat_along_u=True,
                    repeat_along_v=True, uv_scaling=[10,10])

    texture_id, texture_object = _get_texture(env)
    texture_objects.append(texture_object)
    ceiling.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=True, repeat_along_u=True,
                    repeat_along_v=True, uv_scaling=[10,10])

    texture_id, texture_object = _get_texture(env)
    texture_objects.append(texture_object)
    for j in walls:
        j.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=True, repeat_along_u=True,
                    repeat_along_v=True, uv_scaling=[3,10])

    for j in objects:
        texture_id, texture_object = _get_texture(env)
        color = [random() for _ in range(3)]
        texture_objects.append(texture_object)
        for i in j:
            # i.set_color(color)
            i.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, decal_mode=True)

    env.pr.step()

    # Lights
    light_pos = [uniform(-2,2) for _ in range(2)]
    light_pos = light_pos + [uniform(-1.5,-0.5)]
    lights.set_position(light_pos)
    # Camera
    camera_pos = np.array(camera.get_position())
    camera_pos = camera_pos + np.array([uniform(-0.1,0.1) for _ in range(3)])
    camera.set_position(camera_pos.tolist())

    env.pr.step()

    return texture_objects

def remove_textures(floor, walls, ceiling, objects, env):

    [j.remove_texture() for j in walls]
    for j in objects:
        for i in j:
            i.remove_texture()
            i.set_color([176/255, 58/255, 46/255])

    floor.remove_texture()
    ceiling.remove_texture()

    env.pr.step()

def save_sample(rand_active, camera, n_sample, dir):
    if rand_active:
        rgb_img_rand = camera.capture_rgb()
        img_3 = Image.fromarray(np.uint8(rgb_img_rand*255),'RGB')
        _change_dir(dir)
        img_3.save('%s_input0.png'%(n_sample+9613))
    else:
        rgb_img_can = camera.capture_rgb()
        rgb_d_img = camera.capture_depth()
        img_1 = Image.fromarray(np.uint8(rgb_img_can*255),'RGB')
        img_2 = Image.fromarray(np.uint8(rgb_d_img*255))
        _change_dir(dir)
        img_1.save('%s_output0.png'%(n_sample+9613))
        img_2.save('%s_output1.png'%(n_sample+9613))

def _change_dir(dir):
    home_dir = os.path.expanduser('~')
    os.chdir(os.path.join(home_dir,'robotics_drl/data/%s'%(dir)))

def _get_texture(env):
    texture_file = choice(os.listdir('/home/georges/robotics_drl/data/textures/obj_textures/'))
    texture_object, texture_id = env.pr.create_texture(filename='/home/georges/robotics_drl/data/textures/obj_textures/%s'%(texture_file))
    texture_object.set_renderable(0)
    return texture_id, texture_object

def main():
    # if os.path.exists('/home/georges/robotics_drl/data/rcan_data'):
    #     raise Exception('rcan_data file already exists, make sure not overwriting generated data')
    # else:
    #     os.mkdir('/home/georges/robotics_drl/data/rcan_data')
    area = 5**2
    objects_per_m2 = 0.5
    n_samples = 10000
    l_ep = 200

    env = youBotAll(scene_name='scene1.ttt', boundary=5)

    n_objects = int(round(area * objects_per_m2))//2
    env.target_base.set_renderable(0) # Add a cube?
    floor = Shape('floor')
    ceiling = Shape('ceiling')
    lights = Dummy('DefaultLights')
    camera = env.camera_arm

    env.boundary = 2.5

    # target = Shape.create(PrimitiveShape.CUBOID,[0.05,0.05,0.05],orientation=[0,0,0], static=True)
    # target.set_parent(env.target_base)
    # target.set_position([0,0,0],relative_to=env.target_base)
    # target.set_dynamic(1)

    camera_pos = camera.get_position(relative_to=env.mobile_base)

    perm_walls = [Shape('wall%s'%j) for j in range(4)]

    steps = 0
    # Store some position in env then apply domain rand instead of set pos then rand then can
    time_start = time.time()
    for scene_no in range(n_samples//l_ep):

        walls = create_random_scene()
        ceiling.set_position([0,0] + [uniform(2.5,3)])
        walls = walls[0] + walls[1] + perm_walls
        objects_vis = []
        objects_resp = []
        for _ in range(n_objects):
            object_vis, object_resp = add_procedural_object()
            objects_vis.append([object_vis])
            objects_resp.append(object_resp)

            object_vis, object_resp = add_object()
            objects_vis.append(object_vis)
            objects_resp.append(object_resp)

        texture_objects = apply_domain_rand(floor, camera,
                    lights, walls, ceiling, objects_vis, env)
        remove_textures(floor, walls, ceiling, objects_vis, env)
        [j.remove() for j in texture_objects]
        lights.set_position([0,0,0])
        camera.set_position(camera_pos, relative_to=env.mobile_base)

        for _ in range(100): env.pr.step()

        for sample_scene_no in range(l_ep):
            while True:
                x, y, orient = env.rand_bound()
                env.pr.set_configuration_tree(env.config_tree)
                env.mobile_base.set_2d_pose([x, y, orient])
                collision_state = env.mobile_base.assess_collision()
                mobile_orient = env.mobile_base.get_orientation()
                if not collision_state and mobile_orient[0] < 0.2 and mobile_orient[1] < 0.2 :
                    break

            env.pr.step()

            for rand_active in range(2):
                rand_active = not rand_active
                if rand_active:
                    texture_objects = apply_domain_rand(floor, camera,
                                lights, walls, ceiling, objects_vis, env)
                else:
                    lights.set_position([0,0,-0.5])
                    camera.set_position(camera_pos, relative_to=env.mobile_base)
                    remove_textures(floor, walls, ceiling, objects_vis, env)

                save_sample(rand_active, camera, steps, 'rcan_data')

            [j.remove() for j in texture_objects]

            steps += 1

        [j.remove() for j in walls[:-4]]
        for j in objects_vis:
            for i in j:
                i.remove()

        [j.remove() for j in objects_resp]
        walls = []

    env.terminate()

    print(time.time()-time_start)

if __name__ == "__main__":
    main()
