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

def get_CT_Mesh(mat_list_, key_list_=None):
    if key_list_ is None:
        key_list_ = ["facesInner", "facesOuter", "verticesInner", "verticesOuter"]
    tensors = []
    for mat_file, key_ in zip(mat_list_, key_list_):
        mat_data = scipy.io.loadmat(mat_file)[key_].astype(np.float32)
        tensors.append(torch.tensor(mat_data))
    # flip and minus 1 because p3d starts indexing with 0
    tensors[0] = (tensors[0].long() - 1)[:, [2, 1, 0]]
    tensors[1] = (tensors[1].long() - 1)[:, [2, 1, 0]]
    Inner_Mesh = Meshes(verts=[tensors[2]], faces=[tensors[0]])
    Outer_Mesh = Meshes(verts=[tensors[3]], faces=[tensors[1]])
    return Inner_Mesh, Outer_Mesh


def get_US_seg_Mesh(dir_):
    pcds_ = []
    factor = 1000
    for curve in os.listdir(dir_):
        mat_data = scipy.io.loadmat(os.path.join(dir_, curve))
        x_ = torch.tensor(mat_data['curvesVes']['xpint_track'][0, 0].astype(np.float32)).permute(2, 0, 1)
        y_ = torch.tensor(mat_data['curvesVes']['ypint_track'][0, 0].astype(np.float32)).permute(2, 0, 1)
        z_ = torch.tensor(mat_data['curvesVes']['zpint_track'][0, 0].astype(np.float32)).permute(2, 0, 1)
        x_ = x_.reshape(x_.shape[0], -1) * factor
        y_ = y_.reshape(y_.shape[0], -1) * factor
        z_ = z_.reshape(z_.shape[0], -1) * factor
        xyz_ = torch.stack([x_, y_, z_], dim=-1)  # [2, N, 3]
        pcds_.append(xyz_)
    return pcds_


def get_growth_magnitude(fitted_verts_all_frames,
                         Mesh,
                         frame1=2,
                         frame2=0
                         ):
    verts1_ED = fitted_verts_all_frames[frame1]['ED']
    verts1_ES = fitted_verts_all_frames[frame1]['ES']
    verts2_ED = fitted_verts_all_frames[frame2]['ED']
    verts2_ES = fitted_verts_all_frames[frame2]['ES']
    field_ED = verts2_ED - verts1_ED
    field_ES = verts2_ES - verts1_ES
    field_ED = torch.norm(field_ED, dim=1, keepdim=False)
    field_ES = torch.norm(field_ES, dim=1, keepdim=False)
    cloud = pv.PolyData(Mesh.verts_packed().detach().cpu().numpy())
    cloud['growth_ED'] = field_ED.numpy()
    pv.plot(cloud, scalars='growth_ED', cmap='jet', show_bounds=True, cpos='yz', clim=[0, 0.1])
    cloud = pv.PolyData(Mesh.verts_packed().detach().cpu().numpy())
    cloud['growth_ES'] = field_ES.numpy()
    pv.plot(cloud, scalars='growth_ES', cmap='jet', show_bounds=True, cpos='yz', clim=[0, 0.1])


if __name__ == "__main__":
    # CT
    aaa_name = "AAA034"
    key_list = ["facesInner", "facesOuter", "verticesInner", "verticesOuter"]
    mat_list = [os.path.join("./data", aaa_name, "CT mesh", key_ + ".mat") for key_ in key_list]
    # segmentation
    seg_dir_ = os.path.join("./data", aaa_name, "US segmentations")

    Inner_Mesh, Outer_Mesh = get_CT_Mesh(mat_list, key_list)
    IMt = trimesh.Trimesh(vertices=Inner_Mesh.verts_packed().numpy(), faces=Inner_Mesh.faces_packed().numpy())
    OMt = trimesh.Trimesh(vertices=Outer_Mesh.verts_packed().numpy(), faces=Outer_Mesh.faces_packed().numpy())
    save_obj(os.path.join("./data", aaa_name, "outer_mesh.obj"), verts=Outer_Mesh.verts_packed(), faces=Outer_Mesh.faces_packed())
    save_obj(os.path.join("./data", aaa_name, "inner_mesh.obj"), verts=Inner_Mesh.verts_packed(), faces=Inner_Mesh.faces_packed())

    pcds_ = get_US_seg_Mesh(seg_dir_)

    # p = pv.Plotter()
    # p.add_mesh(IMt, opacity=0.5, color='red', show_edges=True)
    # p.add_mesh(OMt, opacity=0.5, color='blue', show_edges=True)
    # p.add_points(pcds_[0][0, ...].numpy(), render_points_as_spheres=True, color='black')
    # # p.add_points(pcds_[-1][1, ...].numpy(), render_points_as_spheres=True, color='yellow')
    # p.add_points(pcds_[1][0, ...].numpy(), render_points_as_spheres=True, color='blue')
    # p.add_points(pcds_[2][0, ...].numpy(), render_points_as_spheres=True, color='red')
    # # p.add_points(pcds_[0][1, ...].numpy(), render_points_as_spheres=True, color='yellow')
    # p.show()

    fitter_Mesh = os.path.join("./data", aaa_name, "fitter_mesh3.obj")
    loss_weighting = {'loss_p0': 1, 'loss_n1': 1,
                      # 'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1,
                      'loss_rigid': 0.25,
                      'axis_reg_loss': 1,
                      'bound_reg_loss': 1
                      }

    AAA_Registration = AAA_Registration(pcds_, fitter_Mesh, loss_weighting, root=os.path.join("./data", aaa_name),
                                        redo_alignment=False,
                                        redo_ghd=True,
                                        num_GHD=12**2,
                                        use_cpu=False,
                                        meta="normal")
    # 1 -> pcd alignment
    fitted_verts_all_frames = getattr(AAA_Registration, "fitted_verts_all_frames")
    Mesh = getattr(AAA_Registration, "AAA_OuterMesh")
    get_growth_magnitude(fitted_verts_all_frames,
                         Mesh,
                         0,
                         2)

"""
align pcd to mesh
GHD 
node displacement tracking

Q: where were the pcd monitored, outer or inner surface?

Q: how did you get the mesh from? s or d

loss_weighting = {'loss_p0': 0.25, 'loss_n1': 0.2,
                  'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1, 'loss_rigid': 25}

python loader.py

sudo docker exec -it wenhao_pvae1 /bin/bash

"""
