from .scene import *
from .utils.vector3 import *
from .materials.diffuse import *
from .materials.emissive import *
from .materials.refractive import *
from .geometry.cuboid import *
from .geometry.plane import *
from .geometry.sphere import *

def cornell_box(screen_size, multi_lights=False):
    Sc = Scene(ambient_color = rgb(0.00, 0.00, 0.00))
    Sc.add_Camera(screen_width = screen_size[0] ,screen_height = screen_size[1], 
        look_from = vec3(278, 278, 800), look_at = vec3(278,278,0), 
        focal_distance= 1., field_of_view= 40)

    green_diffuse=Diffuse(diff_color = rgb(.12, .45, .15))
    red_diffuse=Diffuse(diff_color = rgb(.65, .05, .05))
    white_diffuse=Diffuse(diff_color = rgb(.73, .73, .73))
    emissive_white =Emissive(color = rgb(15., 15., 15.))
    emissive_blue =Emissive(color = rgb(4., 4., 7.))
    emissive_red =Emissive(color = rgb(7., 4., 4.))
    blue_glass =Refractive(n = vec3(1.5 + 0.05e-8j,1.5 +  0.02e-8j,1.5 +  0.j))

    # light
    Sc.add(Plane(material = emissive_white,  center = vec3(213 + 130/2, 554, -227.0 - 105/2), width = 130.0, height = 105.0, u_axis = vec3(1.0, 0.0, 0), v_axis = vec3(0.0, 0, 1.0)), importance_sampled = True)
    if multi_lights:
        Sc.add(Cuboid(material = emissive_blue, center = vec3(50.0, 30.0, -300.0), width = 30,height = 30, length = 30, shadow = False), importance_sampled=True)
        Sc.add(Cuboid(material = emissive_red, center = vec3(50.0, 30.0, -100.0), width = 30,height = 30, length = 30, shadow = False), importance_sampled=True)
        Sc.add(Cuboid(material = emissive_red, center = vec3(500.0, 30.0, -300.0), width = 30,height = 30, length = 30, shadow = False), importance_sampled=True)
        Sc.add(Cuboid(material = emissive_blue, center = vec3(500.0, 30.0, -100.0), width = 30,height = 30, length = 30, shadow = False), importance_sampled=True)

    add_box_walls(Sc)

    cb = Cuboid( material = white_diffuse, center = vec3(182.5, 165, -285-160/2), width = 165,height = 165*2, length = 165, shadow = False)
    cb.rotate(θ = 15, u = vec3(0,1,0))
    Sc.add(cb)

    Sc.add(Sphere( material = blue_glass, center = vec3(370.5, 165/2, -65-185/2), radius = 165/2, shadow = False, max_ray_depth = 3),
					 importance_sampled = True)
    return Sc

def light_slit_box(screen_size):
    Sc = Scene(ambient_color = rgb(0.00, 0.00, 0.00))
    Sc.add_Camera(screen_width = screen_size[0] ,screen_height = screen_size[1], 
        look_from = vec3(278, 278, 800), look_at = vec3(278,278,0), 
        focal_distance= 1., field_of_view= 40)
    add_box_walls(Sc)
    # walls
    white_diffuse=Diffuse(diff_color = rgb(.73, .73, .73))
    Sc.add(Cuboid(material=white_diffuse, center=vec3(350, 555/4, -555/2), width=50, height=555/2, length=555, shadow=True))
    Sc.add(Cuboid(material=white_diffuse, center=vec3(100, 350, -555/2), width=200, height=50, length=555, shadow=True))
    Sc.add(Cuboid(material=white_diffuse, center=vec3(555-100, 100, -300), width=200, height=200, length=100, shadow=True))
    # lights
    emissive_white =Emissive(color = rgb(15., 15., 15.))
    emissive_blue =Emissive(color = rgb(4., 4., 7.))
    emissive_red =Emissive(color = rgb(7., 4., 2.))
    Sc.add(Plane(material=emissive_white, center=vec3(100, 324, -555/2), width=100, height=100, u_axis=vec3(1.0, 0.0, 0), v_axis=vec3(0.0, 0, 1.0)), importance_sampled=True)
    Sc.add(Cuboid(material=emissive_blue, center=vec3(200, 50, -400), width=50, height=50, length=50, shadow=True), importance_sampled=True)
    Sc.add(Cuboid(material=emissive_red, center=vec3(500, 555/4, -100), width=50, height=50, length=50, shadow=True), importance_sampled=True)
    
    return Sc

def add_box_walls(Sc):
    green_diffuse=Diffuse(diff_color = rgb(.12, .45, .15))
    red_diffuse=Diffuse(diff_color = rgb(.65, .05, .05))
    white_diffuse=Diffuse(diff_color = rgb(.73, .73, .73))
    
    Sc.add(Plane(material = white_diffuse,  center = vec3(555/2, 555/2, -555.0), width = 555.0,height = 555.0, u_axis = vec3(0.0, 1.0, 0), v_axis = vec3(1.0, 0, 0.0)))
    Sc.add(Plane(material = green_diffuse,  center = vec3(-0.0, 555/2, -555/2), width = 555.0,height = 555.0,  u_axis = vec3(0.0, 1.0, 0), v_axis = vec3(0.0, 0, -1.0)))
    Sc.add(Plane(material = red_diffuse,    center = vec3(555.0, 555/2, -555/2), width = 555.0,height = 555.0,  u_axis = vec3(0.0, 1.0, 0), v_axis = vec3(0.0, 0, -1.0)))
    Sc.add(Plane(material = white_diffuse,  center = vec3(555/2, 555, -555/2), width = 555.0,height = 555.0,  u_axis = vec3(1.0, 0.0, 0), v_axis = vec3(0.0, 0, -1.0)))
    Sc.add(Plane(material = white_diffuse,  center = vec3(555/2, 0., -555/2), width = 555.0,height = 555.0,  u_axis = vec3(1.0, 0.0, 0), v_axis = vec3(0.0, 0, -1.0)))