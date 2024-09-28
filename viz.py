import numpy as np
import pytorch3d as p3d
import scipy
import torch
import os
from pytorch3d.structures import Meshes
import trimesh
import pyvista as pv
from fitter import AAA_Registration
from pytorch3d.io import load_objs_as_meshes, save_obj


class GHD_visualizer(object):
    def __init__(self, fitted_verts_all_frames, Mesh: Meshes,
                 aligned_pcds_ED,
                 aligned_pcds_ES):
        self.fitted_verts_all_frames = fitted_verts_all_frames
        self.Mesh = Mesh.detach().cpu()
        self.aligned_pcds_ED = aligned_pcds_ED
        self.aligned_pcds_ES= aligned_pcds_ES

    def visualize_GHD_fitting(self, frame=0, label="ED"):
        warped_Mesh = self.Mesh.update_padded(self.fitted_verts_all_frames[frame][label].unsqueeze(0))
        warped_Mesh_trimesh = trimesh.Trimesh(vertices=warped_Mesh.verts_packed().numpy(), faces=warped_Mesh.faces_packed().numpy())
        p = pv.Plotter()
        p.add_mesh(warped_Mesh_trimesh, color='blue', opacity=0.5, show_edges=True)
        if label=="ED":
            pcd_echo = self.aligned_pcds_ED[frame]
        else:
            pcd_echo = self.aligned_pcds_ES[frame]
        p.add_points(pcd_echo.cpu().numpy(), color='red', render_points_as_spheres=True)
        p.show()

    def get_growth_magnitude(self,
                             frame1=2,
                             frame2=0
                             ):
        verts1_ED = self.fitted_verts_all_frames[frame1]['ED']
        verts1_ES = self.fitted_verts_all_frames[frame1]['ES']
        verts2_ED = self.fitted_verts_all_frames[frame2]['ED']
        verts2_ES = self.fitted_verts_all_frames[frame2]['ES']
        field_ED = verts2_ED - verts1_ED
        field_ES = verts2_ES - verts1_ES
        field_ED = torch.norm(field_ED, dim=1, keepdim=False)
        field_ES = torch.norm(field_ES, dim=1, keepdim=False)
        warped_Mesh_ED = self.Mesh.update_padded(self.fitted_verts_all_frames[frame1]["ED"].unsqueeze(0))
        cloud = pv.PolyData(warped_Mesh_ED.verts_packed().detach().cpu().numpy())
        cloud['growth_ED'] = field_ED.numpy()
        pv.plot(cloud, scalars='growth_ED', cmap='jet', show_bounds=True, cpos='yz', point_size=10, clim=[0, 0.3])
        warped_Mesh_ES = self.Mesh.update_padded(self.fitted_verts_all_frames[frame1]["ES"].unsqueeze(0))
        cloud = pv.PolyData(warped_Mesh_ED.verts_packed().detach().cpu().numpy())
        cloud['growth_ES'] = field_ES.numpy()
        pv.plot(cloud, scalars='growth_ES', cmap='jet', show_bounds=True, cpos='yz', point_size=10, clim=[0, 0.3])



