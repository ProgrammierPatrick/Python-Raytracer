from .utils.vector3 import *
from .utils.constants import *
from .utils.colour_functions import *
from .materials.diffuse import *
from .materials.emissive import *
from .ray import *
from .scene import *
import numpy as np
from PIL import Image

def trace_ray(scene : Scene, ray : Ray):
    """ returns tuple (pos, norm) of vec3 objects """
    w = scene.camera.screen_width
    h = scene.camera.screen_height
    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)
    nearest = reduce(np.minimum, distances)

    normal = vec3(np.zeros_like(ray.origin), np.zeros_like(ray.origin), np.zeros_like(ray.origin))

    for (coll, dis, orient) in zip(scene.collider_list, distances, hit_orientation):
        hit_check = (nearest != FARAWAY) & (dis == nearest)
        if np.any(hit_check):
            hitted_rays = ray.extract(hit_check)
            material = coll.assigned_primitive.material
            hit_info = Hit(extract(hit_check, dis), extract(hit_check, orient), material, coll, coll.assigned_primitive)
            hit_info.point = (hitted_rays.origin + hitted_rays.dir * hit_info.distance)
            hit_normal = hit_info.material.get_Normal(hit_info)
            normal = normal.place_values(np.reshape(hit_check, (h,w)), hit_normal)
    
    position = ray.origin + ray.dir * nearest
    return (position, normal)

def trace_ray_material(scene : Scene, ray : Ray):
    """ returns tuple (pos, norm, emission, albedo) of vec3 objects """
    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)
    nearest = reduce(np.minimum, distances)

    normal   = ray.origin.zeros_like()
    emission = ray.origin.zeros_like()
    albedo   = ray.origin.zeros_like()

    for (coll, dis, orient) in zip(scene.collider_list, distances, hit_orientation):
        hit_check = (nearest != FARAWAY) & (dis == nearest)
        if np.any(hit_check):
            hitted_rays = ray.extract(hit_check)
            material = coll.assigned_primitive.material
            hit_info = Hit(extract(hit_check,dis), extract(hit_check,orient), material, coll, coll.assigned_primitive)
            hit_info.point = (hitted_rays.origin + hitted_rays.dir * hit_info.distance)
            hit_normal = hit_info.material.get_Normal(hit_info)
            normal = normal.place_values(hit_check, hit_normal)
            if isinstance(material, Diffuse):
                color = material.diff_texture.get_color(hit_check)
                albedo = albedo.place_values(hit_check, color)
            if isinstance(material, Emissive):
                color = material.texture_color.get_color(hit_check)
                emission = emission.place_values(hit_check, color)
        
    position = ray.origin + ray.dir * nearest
    return (position, normal, emission, albedo)

def sample_sphere(shape):
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    theta = 2 * np.pi * np.random.random_sample(size=shape)
    phi = np.arccos(1 - 2 * np.random.random_sample(size=shape))
    return vec3(np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))

def sample_hemisphere(normal : vec3):
    sample = sample_sphere(normal.shape())
    return vec3.where(sample.dot(normal) < 0, -sample, sample)

def to_image(buffer, dimensions):
    """ Convert numpy array with shape (h, w, channel)  or vec3 into a PIL Image """
    if isinstance(buffer, vec3):
        lin_buffer = np.reshape(buffer.to_array(), (3, dimensions[0] * dimensions[1]))
    else:
        lin_buffer = np.reshape(buffer.transpose(2, 0, 1), (3, dimensions[0] * dimensions[1]))
    srgb_buffer = sRGB_linear_to_sRGB(lin_buffer)
    reshaped_buffer = np.reshape(srgb_buffer, (3, dimensions[1], dimensions[0])).transpose(1, 2, 0)
    return Image.fromarray((np.clip(reshaped_buffer, 0, 1) * 255).astype(np.uint8))

# L(x,w) = Le(x,w) + Int Li(x,w) f(x,w,w') Ndw' dw'
def render_path_tracing(scene, samples = 100, depth = 4):
    """ returns image as numpy array with shape (h,w,3) """
    
    # first sample, can be reused across samples
    ray = scene.camera.get_ray(scene.n)
    (first_pos, first_norm, first_emission, first_albedo) = trace_ray_material(scene, ray)

    w = scene.camera.screen_width
    h = scene.camera.screen_height
    result = first_emission

    for sampleIdx in range(samples):
        factor = first_albedo / samples
        old_pos = first_pos
        old_norm = first_norm
        for i in range(1, depth):
            sample = sample_hemisphere(old_norm)
            factor *= sample.dot(old_norm) * (1 / np.pi)
            (new_pos, new_norm, new_emission, new_albedo) = trace_ray_material(scene, Ray(old_pos + 0.00001 * sample, sample, ray.depth, ray.n, ray.reflections, ray.transmissions, ray.diffuse_reflections))
            result += factor * new_emission
            factor *= new_albedo
            old_pos = new_pos
            old_norm = new_norm
    return result

def sample_hemisphere_cosine(normal : vec3):
    # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#ConcentricSampleDisk
    #print("sample disk")
    x = 2 * np.random.random_sample(size=normal.shape()) - 1
    y = 2 * np.random.random_sample(size=normal.shape()) - 1
    r = np.where(np.abs(x) > np.abs(y), x, y)
    theta = np.where(np.abs(x) > np.abs(y), np.pi / 4 * y / x, np.pi / 2 - np.pi / 4 * x / y)
    del x, y
    d_x = r * np.cos(theta)
    d_y = r * np.sin(theta)
    del r, theta
    # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#CosineSampleHemisphere
    #print("sample 3d")
    d_z = np.sqrt(np.fmax(0, 1 - d_x * d_x - d_y * d_y))
    local_sample = vec3(d_x, d_y, d_z)
    # never colinear trick: CG IntCG, VL01, slide 45
    #print("transform: nevercolinear")
    nevercolinear = vec3(normal.y, normal.z, -normal.x)
    #print("tangent")
    tangent = nevercolinear.cross(normal).normalize()
    #print("tangent type:", type(tangent))
    del nevercolinear
    #print("shape:", tangent.shape())
    #print("bitangent")
    bitangent = normal.cross(tangent)
    #print("result")
    #print("type d_x:", type(d_x), "type tangent", type(tangent), "type d_x * tangent:", type(d_x * tangent), "type tangent * d_x:", type(tangent * d_x))
    result = tangent * d_x
    #print("result shape:", result.shape())
    #print("result += bitan")
    result += bitangent * d_y
    #print("result shape:", result.shape())
    #print("result += normal")
    result += normal * d_z
    #print("result shape:", result.shape())
    return result

def render_path_tracing_MIP_lambert(scene, samples = 100, depth = 4):
    """ returns image as numpy array with shape (h,w,3) """
    
    # first sample, can be reused across samples
    ray = scene.camera.get_ray(scene.n)
    (first_pos, first_norm, first_emission, first_albedo) = trace_ray_material(scene, ray)

    w = scene.camera.screen_width
    h = scene.camera.screen_height
    result = first_emission

    for sampleIdx in range(samples):
        factor = first_albedo / samples
        old_pos = first_pos
        old_norm = first_norm
        for i in range(1, depth):
            sample = sample_hemisphere_cosine(old_norm)
            factor *= (1 / np.pi)
            (new_pos, new_norm, new_emission, new_albedo) = trace_ray_material(scene, Ray(old_pos + 0.00001 * sample, sample, ray.depth, ray.n, ray.reflections, ray.transmissions, ray.diffuse_reflections))
            result += factor * new_emission
            factor *= new_albedo
            old_pos = new_pos
            old_norm = new_norm
    return result