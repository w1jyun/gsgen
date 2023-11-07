import trimesh
import numpy as np
import platform
import os
import pyrender
import matplotlib.pyplot as plt
import torch

class ModelRenderer():
    def __init__(self, path, center, scale):
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        model_w = trimesh.load('w_'+path, force='mesh', file_type='glb')
        model_b = trimesh.load('b_'+path, force='mesh', file_type='glb')

        # normalize mesh
        # model = trimesh.load(path, force='mesh', skip_material=True, process=False)
        # scale = 1.0 / np.array(model.bounds[1] - model.bounds[0]).max()
        # center = np.array(model.bounds[1] + model.bounds[0]) / (-2)
        center = -center[0]
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

    def render(self, poses, cam_infos):
        colors = []
        for pos, cam in zip(poses, cam_infos):
            r = pyrender.OffscreenRenderer(512, 512)
            pos = pos.detach().cpu().numpy()
            # y = [0, 1, 0] # up
            # z = c2w[:3, 2] * (-1) # forward
            # x = np.cross(y, z) # right
            # center = c2w[:3, 3]
            c2w = np.identity(4, dtype=np.float32)
            # print('right: ', pos[:3, 0] )
            # print('up: ', pos[:3, 1] )
            # print('forward: ', pos[:3, 2] )
            # print('center: ', pos[:3, 3] )
            c2w[:3, 1] = [-pos[:3, 1][0],-pos[:3, 1][2],-pos[:3, 1][1]]
            c2w[:3, 2] = [-pos[:3, 2][0],-pos[:3, 2][2],-pos[:3, 2][1]]
            c2w[:3, 0] = np.cross(c2w[:3, 1], c2w[:3, 2])
            c2w[:3, 0] /= -np.linalg.norm(c2w[:3, 0])
            c2w[:3, 3] = [pos[:3, 3][0],pos[:3, 3][2],pos[:3, 3][1]]
            fov = cam.getFov()
            scene = self.scene
            # set camera pose
            camera = pyrender.PerspectiveCamera(yfov=(fov), znear=0.1, zfar=10000, aspectRatio=1)
            self.scene.add(camera, pose=np.eye(4))
            cam_node = None
            for node in scene.get_nodes():
                if (node.camera != None):
                    cam_node = node
                    scene.set_pose(node, pose=np.reshape(c2w, (4, 4)))
            color, depth = r.render(scene)
            colors.append(color)
            # plt.imsave('render_.png', color)
            # plt.close()
            r.delete()
            self.scene.remove_node(cam_node)
            del depth, cam_node, camera
        return colors