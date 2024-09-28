import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))

from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes
# from .rigid_deform import RigidLoss
from .mesh_loss import Rigid_Loss
from .diceloss import BinaryDiceLoss, BinaryDiceLoss_Weighted

from ops.graph_operators import Laplacain, Normal_consistence

from ops.torch_warping import feature_align_with_turbulence
from torch_geometric.utils import to_undirected
import pyvista as pv


class Mesh_loss(nn.Module):
    def __init__(self, mesh_std: Meshes, sample_num=5000, resolution=128, length=2, ):
        super(Mesh_loss, self).__init__()
        self.sample_num = sample_num
        self.mesh_std = mesh_std

        cotweight_std, _ = cot_laplacian(mesh_std.verts_packed(), mesh_std.faces_packed())  # laplace matrix
        
        self.connection_std = cotweight_std.coalesce().indices()
        self.cotweight_std = cotweight_std.coalesce().values()


        self.connection_std, self.cotweight_std = to_undirected(self.connection_std, edge_attr=self.cotweight_std)
        
        self.rigidloss = Rigid_Loss(mesh_std)
        self.lap = Laplacain()
        self.normal_consistence = Normal_consistence()

        self.edge_lenth = torch.norm(self.mesh_std.verts_packed().index_select(0,self.connection_std[0]) - self.mesh_std.verts_packed().index_select(0,self.connection_std[1]),dim=-1).mean()

        self.resolution = resolution
        self.length = length
        # init with a mesh to give 2 things: vert structure to use loss_rigid and edge_length for edge_loss

        # loss modules
        self.dice_loss = BinaryDiceLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.dice_loss_attention = BinaryDiceLoss_Weighted(weights_normalize=False)  # conservative: not using weights_normalization
    
    def forward(self, Mesh, echo_pcd: torch.Tensor, pcd_vector, loss_list: dict, rigid_static_Mesh, aligned_pcd_norm):
        loss_dict = {}
        device = Mesh.device
        echo_pcd = echo_pcd.to(device)
        pcd_vector = pcd_vector[0, :].view(1,3).to(device)

        sample_pcd, sample_norm = sample_points_from_meshes(Mesh, self.sample_num, return_normals=True)
        sample_pcd, sample_norm = point_filter(sample_pcd.squeeze(0), echo_pcd, pcd_vector, sample_norm.squeeze(0))
        loss_p0, loss_n1 = chamfer_distance(x=sample_pcd.unsqueeze(0), y=echo_pcd.unsqueeze(0), x_normals=sample_norm.unsqueeze(0),
                                            y_normals=aligned_pcd_norm.unsqueeze(0))

        if 'loss_p0' in loss_list:
            loss_dict['loss_p0'] = loss_p0 if not torch.isnan(loss_p0) else torch.Tensor([0.0]).to(self.device)

        if 'loss_n1' in loss_list:
            loss_dict['loss_n1'] = loss_n1 if not torch.isnan(loss_n1) else torch.Tensor([0.0]).to(self.device)


        if 'loss_laplacian' in loss_list:
        
            laplacain_vect = mesh_laplacian_smoothing(Mesh, method="cot")
    
            loss_dict['loss_laplacian'] = laplacain_vect if not torch.isnan(laplacain_vect) else torch.Tensor([0.0]).to(self.device)


        if 'loss_edge' in loss_list:
            mesh_edge_loss_item = mesh_edge_loss(Mesh,self.edge_lenth.to(Mesh.device))
            loss_dict['loss_edge'] = mesh_edge_loss_item if not torch.isnan(mesh_edge_loss_item) else torch.Tensor([0.0]).to(self.device)

        if 'loss_consistency' in loss_list:
            mesh_normal_consistency_item = mesh_normal_consistency(Mesh)
            loss_dict['loss_consistency'] = mesh_normal_consistency_item if not torch.isnan(mesh_normal_consistency_item) else torch.Tensor([0.0]).to(self.device)
            
        if 'loss_rigid' in loss_list:
            verts_scr = Mesh.verts_padded()
            verts_std = self.mesh_std.verts_packed()

            loss_dict['loss_rigid'] = torch.zeros_like(verts_std[:,:1])

            B = 1
            for i in range(B):
                rigid_i = self.rigidloss.forward(verts_scr[i])
                loss_dict['loss_rigid'] += rigid_i if not torch.isnan(rigid_i) else torch.Tensor([0.0]).to(self.device)
            loss_dict['loss_rigid'] = loss_dict['loss_rigid'].mean()/B

        if "axis_reg_loss" in loss_list:
            axis_reg_loss = axis_regularization(rigid_static_Mesh.to(device), Mesh, pcd_vector)
            loss_dict['axis_reg_loss'] = axis_reg_loss

        if "bound_reg_loss" in loss_list:
            bound_reg_loss = boundary_regularization(rigid_static_Mesh.to(device), Mesh, pcd_vector)
            loss_dict['bound_reg_loss'] = bound_reg_loss
        return loss_dict


def axis_regularization(Mesh_, warped_Mesh_, pca_Mesh_):
    loss_ = nn.MSELoss(reduction='sum')
    projection_Mesh_ = torch.matmul(Mesh_.verts_packed().to(warped_Mesh_.device), pca_Mesh_[0, :].view(1, 3).transpose(0, 1))
    projection_warped_Mesh_ = torch.matmul(warped_Mesh_.verts_packed().to(warped_Mesh_.device), pca_Mesh_[0, :].view(1, 3).transpose(0, 1))
    axis_reg_loss = loss_(projection_warped_Mesh_ - projection_Mesh_, torch.zeros_like(projection_warped_Mesh_))
    return axis_reg_loss


def boundary_regularization(Mesh_, warped_Mesh_, pca_Mesh_, percentage=0.05):
    loss_br = nn.MSELoss(reduction='sum')
    projection_Mesh_ = torch.matmul(Mesh_.verts_packed().to(warped_Mesh_.device),
                                    pca_Mesh_[0, :].view(1, 3).transpose(0, 1))
    num_nodes_br = int(projection_Mesh_.numel() * percentage)
    sorted_vals, sorted_indices = torch.sort(projection_Mesh_.view(-1))
    indices = torch.cat((sorted_indices[:num_nodes_br], sorted_indices[-num_nodes_br:]))
    boundary_reg_loss = loss_br(Mesh_.verts_packed()[indices], warped_Mesh_.verts_packed()[indices])
    return boundary_reg_loss


def point_filter(pcd1, pcd2, vector, norm1=None):
    """

    pcd: [N, 3]
    vector: [1, 3]
    pcd1: pcd to filter
    pcd2: pcd for measure
    """
    project1 = torch.matmul(pcd1 - torch.mean(pcd2, dim=0, keepdim=True), vector.transpose(0, 1))
    project2 = torch.matmul(pcd2 - torch.mean(pcd2, dim=0, keepdim=True), vector.transpose(0, 1))
    # p = pv.Plotter()
    # p.add_points(pcd1.detach().cpu().numpy(), color='red')
    # p.add_points(pcd2.detach().cpu().numpy(), color='blue')
    # arrow1 = pv.Arrow(torch.mean(pcd2, dim=0, keepdim=True).detach().cpu().numpy(), vector, scale=5)
    # p.add_mesh(arrow1, color='yellow')
    # p.show()
    max_ = torch.max(project2)
    min_ = torch.min(project2)
    indices = torch.where((project1 >= min_) & (project1 <= max_))[0]
    if norm1 is None:
        return pcd1[indices, :]
    else:
        return pcd1[indices, :], norm1[indices, :]

if __name__ == "__main__":
    from pytorch3d.io import load_objs_as_meshes
    B = 2
    device = torch.device("cpu")
    mesh_std = load_objs_as_meshes(['/home/yihao/data/PINN_GCN/data/canonical.obj'],device=device)[0]
    N = mesh_std.verts_packed().shape[0]
    meshes_scr = load_objs_as_meshes(['/home/yihao/data/PINN_GCN/data/canonical.obj']*B,device=device)
    meshes_scr = meshes_scr.offset_verts(torch.randn(B*N,3).to(device))
    meshes_trg = meshes_scr.offset_verts(torch.randn(B*N,3).to(device))

    gt3d = torch.randint(0,2,(B,2,128,128,128)).float().to(device)
    #ico_sphere(3, device)
    mesh_loss = Mesh_loss(mesh_std,50)
    loss_dict = mesh_loss(meshes_scr, meshes_trg, ['loss_p0','loss_n1','loss_consistency','loss_laplacian','loss_edge','loss_seg_align','loss_rigid'],gt3d)
    print(loss_dict)