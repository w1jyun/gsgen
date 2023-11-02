import trimesh
import numpy as np
import platform
import os
import pyrender

# def createDirectory(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print("Error: Failed to create the directory.")

class ModelRenderer():
    def __init__(self, path):
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        model_w = trimesh.load('w_'+path, force='mesh', file_type='glb')
        model_b = trimesh.load('b_'+path, force='mesh', file_type='glb')

        # normalize mesh
        model = trimesh.load(path, force='mesh', skip_material=True, process=False)
        scale = 0.8 / np.array(model.bounds[1] - model.bounds[0]).max()
        center = np.array(model.bounds[1] + model.bounds[0]) / (-2)
        
        model_w.apply_transform(trimesh.transformations.translation_matrix(center))
        model_w.apply_transform(trimesh.transformations.scale_matrix(scale))
        model_b.apply_transform(trimesh.transformations.translation_matrix(center))
        model_b.apply_transform(trimesh.transformations.scale_matrix(scale))

        mesh_w = pyrender.Mesh.from_trimesh(model_w, smooth=False)
        mesh_b = pyrender.Mesh.from_trimesh(model_b, smooth=False)
        
        scene = pyrender.Scene(ambient_light=(1, 1, 1), bg_color=(0.0, 0.0, 1.0, 1.0))
        scene.add(mesh_w, name='white', pose= np.eye(4))
        scene.add(mesh_b, name='black', pose= np.eye(4))

        self.scene = scene
        for node in scene.get_nodes():
            if (node.mesh != None):
                if node.name == 'white':
                    node.mesh.primitives[0].material.emissiveFactor = [1.0, 1.0, 1.0, 1.0]
                else:
                    node.mesh.primitives[0].material.emissiveFactor = [0.0, 0.0, 0.0, 1.0]
        # yfov, znear=0.05, zfar=None, aspectRatio=None, name=None

        # createDirectory('./control/')
        # self.path = './control/'

    def render(self, poses, cam_infos):
        colors = []
        for pos, cam in zip(poses, cam_infos):
            r = pyrender.OffscreenRenderer(128, 128)
            position = pos.detach().cpu().numpy()
            position = np.insert(position, 3, [0,0,0,1], axis = 0)
            fov = cam.getFov()
            scene = self.scene
            # set camera pose
            camera = pyrender.PerspectiveCamera(yfov=(fov), znear=0.1, zfar=10000, aspectRatio=1)
            self.scene.add(camera, pose=np.eye(4))
            cam_node = None
            for node in scene.get_nodes():
                if (node.camera != None):
                    cam_node = node
                    scene.set_pose(node, pose=np.reshape(position, (4, 4)))
            color, depth = r.render(scene)
            colors.append(color)

            r.delete()
            self.scene.remove_node(cam_node)
            del position, depth, cam_node, camera

        return colors
    
# plt.imsave('/control/%3d.png'%index, color)
# plt.close()
# def safe_normalize(x, eps=1e-20):
#     return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# def circle_poses(device, radius=torch.tensor([200.]), theta=torch.tensor([60]), phi=torch.tensor([0]), return_dirs=True, angle_overhead=30, angle_front=60):

#     theta = theta / 180 * np.pi
#     phi = phi / 180 * np.pi
#     angle_overhead = angle_overhead / 180 * np.pi
#     angle_front = angle_front / 180 * np.pi

#     centers = torch.stack([
#         radius * torch.sin(theta) * torch.sin(phi),
#         radius * torch.cos(theta),
#         radius * torch.sin(theta) * torch.cos(phi),
#     ], dim=-1) # [B, 3]

#     # lookat
#     forward_vector = safe_normalize(centers)
#     up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

#     poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers


#     return poses

# modelRenderer = ModelRenderer('turt.glb', 20)

# import random

# def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=True, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5):
#     ''' generate random poses from an orbit camera
#     Args:
#         size: batch size of generated poses.
#         device: where to allocate the output.
#         radius: camera radius
#         theta_range: [min, max], should be in [0, pi]
#         phi_range: [min, max], should be in [0, 2 * pi]
#     Return:
#         poses: [size, 4, 4]
#     '''

#     theta_range = np.array(theta_range) / 180 * np.pi
#     phi_range = np.array(phi_range) / 180 * np.pi
#     angle_overhead = angle_overhead / 180 * np.pi
#     angle_front = angle_front / 180 * np.pi

#     radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

#     if random.random() < uniform_sphere_rate:
#         unit_centers = F.normalize(
#             torch.stack([
#                 torch.randn(size, device=device),
#                 torch.abs(torch.randn(size, device=device)),
#                 torch.randn(size, device=device),
#             ], dim=-1), p=2, dim=1
#         )
#         thetas = torch.acos(unit_centers[:,1])
#         phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
#         phis[phis < 0] += 2 * np.pi
#         centers = unit_centers * radius.unsqueeze(-1)
#     else:
#         thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
#         phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
#         phis[phis < 0] += 2 * np.pi

#         centers = torch.stack([
#             radius * torch.sin(thetas) * torch.sin(phis),
#             radius * torch.cos(thetas),
#             radius * torch.sin(thetas) * torch.cos(phis),
#         ], dim=-1) # [B, 3]

#     targets = 0

#     # jitters

#     # lookat
#     forward_vector = safe_normalize(centers - targets)
#     up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

#     up_noise = 0

#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

#     poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers

#     if return_dirs:
#         dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
#     else:
#         dirs = None

#     # back to degree
#     thetas = thetas / np.pi * 180

#     phis = phis / np.pi * 180
#     return poses, dirs, thetas, phis, radius

# @torch.no_grad()
# def get_view_direction(thetas, phis, overhead, front):
#     #                   phis: [B,];          thetas: [B,]
#     # front = 0             [-front/2, front/2)
#     # side (cam left) = 1   [front/2, 180-front/2)
#     # back = 2              [180-front/2, 180+front/2)
#     # side (cam right) = 3  [180+front/2, 360-front/2)
#     # top = 4               [0, overhead]
#     # bottom = 5            [180-overhead, 180]
#     res = torch.zeros(thetas.shape[0], dtype=torch.long)
#     # first determine by phis
#     phis = phis % (2 * np.pi)
#     res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
#     res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
#     res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
#     res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
#     # override by thetas
#     res[thetas <= overhead] = 4
#     res[thetas >= (np.pi - overhead)] = 5
#     return res

# import torch.nn.functional as F


# control = []
# for i in range(1000):
#     poses, dirs, thetas, phis, radius = rand_poses(1, 'cpu')
#     modelRenderer.render(poses, dirs, thetas, phis, radius, i)

# import random
# file_path = random.choice(os.listdir('control'))
# print('./control/' + file_path)
# data = np.load('./control/' + file_path, allow_pickle=True)
# poses, dirs, thetas, phis, radius, color = data
# print(poses)
# print(dirs)
# print(thetas)
# print(phis)
# print(radius)
# print(color)