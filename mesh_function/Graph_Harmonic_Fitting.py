import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import cubify, cot_laplacian, sample_points_from_meshes, knn_points, knn_gather, norm_laplacian

from torch_geometric.utils import degree, to_undirected, to_dense_adj, get_laplacian, add_self_loops
from torch_geometric.data import Data
# from torch_geometric.transforms import gdc
from torch_scatter import scatter

import numpy as np

import trimesh

from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix


# from data_process.dataset_real_scaling import MMWHS_dataset, ACDC_dataset, read_nii_into_world_coord, mask_img3d
from ops.graph_operators import NativeFeaturePropagation, LaplacianSmoothing
from losses import Mesh_loss

from tqdm import tqdm

# from probreg import cpd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from losses.rigid_deform import RigidLoss
from ops.graph_operators import Laplacain, Normal_consistence

from torch_geometric.utils import to_undirected

# import cupy as cp

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

def mix_laplacian(base_shape: Meshes):

    device = base_shape.device

    cotweight, _ = cot_laplacian(base_shape.verts_packed(), base_shape.faces_packed())  # n: cot_laplacian: diag should be zeros

    connection = cotweight.coalesce().indices()

    cotweight_value = cotweight.coalesce().values()

    connection, cotweight_value = to_undirected(connection, edge_attr=cotweight_value)  # n: undirect graph (edge a -> b and b -> a)

    normweight = norm_laplacian(base_shape.verts_packed(), base_shape.edges_packed())  # n: norm_laplacian is lap normalized with normalized distance

    # degree is a diagonal matrix
    connection_1 = normweight.coalesce().indices()
    normweight_value = normweight.coalesce().values()

    _, normweight = to_undirected(connection_1, edge_attr=normweight_value)

    normweight_mean = scatter(normweight, connection[0], dim=0, reduce='add')

    # nortrace = normweight_mean.mean()

    # normweight_mean = normweight_mean / nortrace

    cotweight_mean = scatter(cotweight_value, connection[0], dim=0, reduce='add')

    normweight_value = normweight_value / normweight_mean[connection[0]]

    cotweight_value = cotweight_value / cotweight_mean[connection[0]]
    

    norlap =  add_self_loops(connection,edge_attr=normweight_value,fill_value='add', num_nodes=base_shape.verts_packed().shape[0])


    norlap_adj = add_self_loops(connection,edge_attr=normweight_value,fill_value=0, num_nodes=base_shape.verts_packed().shape[0])

    

    norlap = torch.sparse_coo_tensor(indices=norlap[0], values=norlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap_adj = torch.sparse_coo_tensor(indices=norlap_adj[0], values=norlap_adj[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap = norlap - 2*norlap_adj

    

    norlap_np = coo_matrix((norlap.coalesce().values().cpu().numpy(), (norlap.coalesce().indices()[0].cpu().numpy(), norlap.coalesce().indices()[1].cpu().numpy())))

    cotlap =  add_self_loops(connection,edge_attr=-cotweight_value,fill_value=1+1e-6, num_nodes=base_shape.verts_packed().shape[0])
    cotlap = torch.sparse_coo_tensor(indices=cotlap[0], values=cotlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    cotlap_np = coo_matrix((cotlap.coalesce().values().cpu().numpy(), (cotlap.coalesce().indices()[0].cpu().numpy(), cotlap.coalesce().indices()[1].cpu().numpy())))




    # standerdegree = degree(connection[0], num_nodes=meshes_scr_lv.verts_packed().shape[0], dtype=torch.float32)
    standerlap = get_laplacian(connection, num_nodes = base_shape.verts_packed().shape[0], dtype=torch.float32, normalization='sym')
    standerlap = torch.sparse_coo_tensor(indices=standerlap[0], values=standerlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    standerlap_np = coo_matrix((standerlap.coalesce().values().cpu().numpy(), (standerlap.coalesce().indices()[0].cpu().numpy(), standerlap.coalesce().indices()[1].cpu().numpy())))


    return cotlap_np, norlap_np, standerlap_np

# meshes_scr_lv =  load_objs_as_meshes(["/home/yihao/data/ParaHearts/data/canonical_worldcoord.obj"]).to('cuda:0')

# cotlap_np, norlap_np, standerlap_np = mix_laplacian(meshes_scr_lv)

# print(cotlap_np.shape, norlap_np, standerlap_np)


class Graph_Harmonic_Deform(nn.Module):

    def __init__(self, base_shape: Meshes, num_Basis = 6*6+1, mix_lap_weight= [1,0.1,0.1]):
        super(Graph_Harmonic_Deform, self).__init__()

        self.device = base_shape.device

        self.base_shape = base_shape
    
        self.cotlap_np, self.norlap_np, self.standerlap_np = mix_laplacian(base_shape)  # n: get 3 different laps

        self.mix_lap = mix_lap_weight[0]*self.cotlap_np + mix_lap_weight[1]*self.norlap_np + mix_lap_weight[2]*self.standerlap_np  # n: liear mixture

        self.GBH_eigval, self.GBH_eigvec = eigsh(self.mix_lap, k=num_Basis, which='SM')  # n: get eigs

        self.GBH_eigvec = torch.from_numpy(self.GBH_eigvec).to(base_shape.device).float()  # (V, num_Basis)

        self.GBH_eigval = torch.from_numpy(self.GBH_eigval).to(base_shape.device).float().unsqueeze(0) # (1, num_Basis)

        self.deformation_param = nn.Parameter(torch.zeros((num_Basis, 3), dtype=torch.float32, device=base_shape.device), requires_grad=True)

        self.reset_affine_param()
    
    def to(self, device):
        self.device = device
        self.base_shape = self.base_shape.to(device)
        self.GBH_eigvec = self.GBH_eigvec.to(device)
        self.GBH_eigval = self.GBH_eigval.to(device)
        self.deformation_param = nn.Parameter(self.deformation_param.to(device))
        self.R = nn.Parameter(self.R.to(device))
        self.s = nn.Parameter(self.s.to(device))
        self.T = nn.Parameter(self.T.to(device))
        return self

    def reset_affine_param(self):
        self.R = nn.Parameter(torch.zeros(1,3, device=self.device))
        self.s = nn.Parameter(torch.tensor([1.], device=self.device).unsqueeze(0))
        self.T = nn.Parameter(torch.zeros(1,3, device=self.device))

    def ghb_coefficient_recover(self, GHB_coefficient):
        assert GHB_coefficient.shape[0] == self.GBH_eigvec.shape[-1]
        try:
            n = GHB_coefficient.shape[1]
        except:
            n = 1
            GHB_coefficient = GHB_coefficient.unsqueeze(-1)

        return self.GBH_eigvec.matmul(GHB_coefficient)

    def ghb_coefficient_recover_radial(self, GHB_coefficient_radial, axis_vector, centroid=None):
        assert GHB_coefficient_radial.shape[0] == self.GBH_eigvec.shape[-1]
        # try:
        #     n = GHB_coefficient.shape[1]
        # except:
        #     n = 1
        #     GHB_coefficient = GHB_coefficient.unsqueeze(-1)
        if centroid is None:
            centroid = torch.norm(self.base_shape.verts_packed(), dim=0, keepdim=True)
        else:
            centroid = centroid.view(1, 3)
        v1 = norm_vectors(self.base_shape.verts_packed() - centroid)
        v2 = torch.matmul(torch.sum(v1 * norm_vectors(axis_vector.view(1, 3)), dim=1, keepdim=True), axis_vector.view(1, 3))
        v_radial = norm_vectors(v2 - v1)  # [N, 3]
        deformation_cartesian = self.GBH_eigvec.matmul(GHB_coefficient_radial) * v_radial
        return deformation_cartesian

    def ghb_coefficient_recover_normal(self, GHB_coefficient_radial, output_shape):
        assert GHB_coefficient_radial.shape[0] == self.GBH_eigvec.shape[-1]
        normal = output_shape.verts_normals_packed()
        deformation_normal = self.GBH_eigvec.matmul(GHB_coefficient_radial) * normal
        output_shape = output_shape.offset_verts(deformation_normal)
        return output_shape
    
    def project_to_ghb_eig(self, input_shape):
        assert input_shape.shape[0] == self.GBH_eigvec.shape[0]
        d = input_shape.shape[-1]
        return self.GBH_eigvec.transpose(-1,-2).matmul(input_shape)
    
    def forward(self, GHB_coefficient=None):

        if GHB_coefficient is None:
            GHB_coefficient = self.deformation_param

        deformation = self.ghb_coefficient_recover(GHB_coefficient)  # n: recover laplacian

        output_shape = self.base_shape.offset_verts(deformation)  # n: x, y, and z coefficients are seperated. so we have (basis, 3)

        R_matrix = axis_angle_to_matrix(self.R)

        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1,-2)*self.s + self.T).float())  # n: not solved: rigid scaling, rot, and trans

        return output_shape

    def forward_radial(self, GHB_coefficient, radial_GHB_coefficient, axis_vector, centroid):
        # assert self.s == torch.Tensor([1.]).to(self.s.device)


        deformation = self.ghb_coefficient_recover(GHB_coefficient)  # n: recover laplacian

        deformation_cartesian = self.ghb_coefficient_recover_radial(radial_GHB_coefficient, axis_vector, centroid)

        output_shape = self.base_shape.offset_verts(deformation+deformation_cartesian)  # n: x, y, and z coefficients are seperated. so we have (basis, 3)

        R_matrix = axis_angle_to_matrix(self.R)

        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1,-2)*self.s + self.T).float())  # n: not solved: rigid scaling, rot, and trans

        return output_shape

    def forward_normal(self, GHB_coefficient, radial_GHB_coefficient, mapping_coefficient):
        # assert self.s == torch.Tensor([1.]).to(self.s.device)


        deformation = self.ghb_coefficient_recover(GHB_coefficient)  # n: recover laplacian

        output_shape = self.base_shape.offset_verts(deformation * mapping_coefficient)


        # n: x, y, and z coefficients are seperated. so we have (basis, 3)

        R_matrix = axis_angle_to_matrix(self.R)

        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1,-2)*self.s + self.T).float())  # n: not solved: rigid scaling, rot, and trans

        output_shape = self.ghb_coefficient_recover_normal(radial_GHB_coefficient, output_shape)

        rigid_static_Mesh = self.base_shape.to(output_shape.device).update_padded((self.base_shape.to(output_shape.device).verts_padded() @ R_matrix.transpose(-1,-2)*self.s + self.T).float())

        return output_shape, rigid_static_Mesh

def rigid_registration(point_cloud_scr, point_cloud_target, sample_num=2000, iter_num=2, tf_init_params={}, update_scale=True):
    pt_scr = point_cloud_scr[np.random.choice(len(point_cloud_scr), sample_num, replace=True)]
    pt_trg= point_cloud_target[np.random.choice(len(point_cloud_target), sample_num, replace=True)]
    param_dict =   tf_init_params
    for _ in range(iter_num):
        rgd_cpd = cpd.RigidCPD(pt_scr, tf_init_params=param_dict, update_scale=update_scale, use_cuda=False)
        tf_param, _, _ = rgd_cpd.registration(pt_trg)
        R, s, t =tf_param.rot, tf_param.scale, tf_param.t
        param_dict = {'rot':R, 'scale':s, 't':t}

    return R, s, t

class Biventricle_navigation():
    def __init__(self, canonical_shape_bi:trimesh.Trimesh, canonical_shape_lv:trimesh.Trimesh, sampling_mode = ['inner','surface'], sample_num=5000):
        super(Biventricle_navigation, self).__init__()

        self.canonical_shape_bi = canonical_shape_bi

        self.canonical_shape_lv = canonical_shape_lv

        self.point_cloud_sampling(sample_num=sample_num, sampling_mode = sampling_mode)

        
    def point_cloud_sampling(self, sample_num=5000, sampling_mode = ['inner','surface']):

        if sampling_mode[0] == 'inner':
            self.point_cloud_scr_bi = trimesh.sample.volume_mesh(self.canonical_shape_bi, sample_num)
        elif sampling_mode[0] == 'surface':
            self.point_cloud_scr_bi = trimesh.sample.sample_surface(self.canonical_shape_bi, sample_num)[0]

        if sampling_mode[1] == 'inner':
            self.point_cloud_scr_lv = trimesh.sample.volume_mesh(self.canonical_shape_lv, sample_num)
        elif sampling_mode[1] == 'surface':
            self.point_cloud_scr_lv = trimesh.sample.sample_surface(self.canonical_shape_lv, sample_num)[0]
    
    def rigid_registration_bilv(self, point_cloud_target_bi, point_cloud_target_lv, sample_num=1000, iter_num_biv=2, iter_num_lv=2, init_deformation_param=None):


        if type(point_cloud_target_bi) == torch.Tensor:
            point_cloud_target_bi = point_cloud_target_bi.cpu().numpy()
        if type(point_cloud_target_lv) == torch.Tensor:
            point_cloud_target_lv = point_cloud_target_lv.cpu().numpy()

        if init_deformation_param is None:
            param_dict = {}
        else:
            param_dict = init_deformation_param



        R, s, T = rigid_registration(self.point_cloud_scr_bi, point_cloud_target_bi, iter_num=iter_num_biv, update_scale=False, sample_num=sample_num, tf_init_params=param_dict)
        
        param_dict = {'rot':R, 'scale':1, 't':T}
        R, s, T = rigid_registration(self.point_cloud_scr_lv, point_cloud_target_lv, iter_num=iter_num_lv, update_scale=True, tf_init_params=param_dict, sample_num=sample_num)

        return R, s, T

    
class GHB_Fitting(Graph_Harmonic_Deform):

    def __init__(self, canonical_shape_bi_path, canonical_shape_lv_path, device='cuda:1', sample_num=1000, sampling_mode = ['inner','surface'], num_Basis = 6*6+1):

        self.device = device
        
        base_shape = load_objs_as_meshes([canonical_shape_lv_path], device=device)

        self.canonical_shape_bi = trimesh.load(canonical_shape_bi_path)

        self.canonical_shape_lv = trimesh.load(canonical_shape_lv_path)

        super(GHB_Fitting, self).__init__(base_shape = base_shape, num_Basis = num_Basis)




        
        self.smoother = LaplacianSmoothing()

        self.reg_sample_num= sample_num

        self.fitting_sample_num = 10000

        self.iter_num_biv=2

        self.iter_num_lv=2

        self.fitting_loss = Mesh_loss(self.base_shape, sample_num=self.fitting_sample_num)

        self.biventricle_navig  = Biventricle_navigation(self.canonical_shape_bi, self.canonical_shape_lv, sample_num=sample_num, sampling_mode=sampling_mode)

        self.weight_dict = {'loss_p0': 1.0, 'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1, 'loss_rigid': 1.}
    
    def global_register(self, point_cloud_target_bi, point_cloud_target_lv):

        init_deformation_param = {'rot': axis_angle_to_matrix(self.R.squeeze(0)).detach().cpu().numpy(), 'scale':self.s.detach().cpu().numpy().squeeze(0)[0], 't':self.T.detach().cpu().numpy().squeeze(0)}

        # print('init_deformation_param: ', init_deformation_param)

        R, s, T = self.biventricle_navig.rigid_registration_bilv(point_cloud_target_bi, point_cloud_target_lv, sample_num=self.reg_sample_num, iter_num_biv=self.iter_num_biv, iter_num_lv=self.iter_num_lv, init_deformation_param=init_deformation_param)
        
        self.R = nn.Parameter(matrix_to_axis_angle(torch.from_numpy(R).float().to(self.device)).unsqueeze(0))
        self.s = nn.Parameter(torch.tensor([s], device=self.device).unsqueeze(0))
        self.T = nn.Parameter(torch.from_numpy(T).float().to(self.device).unsqueeze(0))

        return self.R, self.s, self.T
    
    def ghb_fitting(self, target, iter_num= 5000):

        
        optimizer = torch.optim.Adam([self.deformation_param, self.s, self.T], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_num, eta_min=1e-5)

        pbar = tqdm(range(iter_num), mininterval=30, ncols=100)
        
        for i in pbar:

            current_shpae = self.forward()
            optimizer.zero_grad()

            loss_dict = self.fitting_loss(current_shpae, target, self.weight_dict, B=1)
            loss = torch.zeros(1, device=self.device)
            for key in loss_dict:
                # loss_dict[key].backward(retain_graph=True)
                # loss = loss*torch.pow(loss_dict[key], self.weight_dict[key])
                loss += loss_dict[key]*self.weight_dict[key]
            loss.backward()

            if i%30 == 0:
                pbar.set_description("chamfer loss: %.6f" % loss_dict['loss_p0'].item())
            
            optimizer.step()
            scheduler.step()
            
        return self.deformation_param, self.s, loss_dict['loss_p0'].item()


class GHB_DENSE_Fitting(GHB_Fitting):

    def __init__(self, canonical_shape_bi_path, canonical_shape_lv_path, device='cuda:1', num_Basis = 6*6+1, sample_num=2000, sampling_mode = ['inner','surface']):


        super(GHB_DENSE_Fitting, self).__init__(canonical_shape_bi_path,canonical_shape_lv_path, device, sample_num = sample_num, sampling_mode = ['inner','surface'], num_Basis = num_Basis)

        self.fitting_sample_num = 20000

        self.weight_dict = {'loss_p0': 1.0, 'loss_n1': 0.5, 'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1, 'loss_rigid': 3.}

        self.fitting_loss = Mesh_loss(self.base_shape, sample_num=self.fitting_sample_num)


    def GHB_fitting_from_Tensor(self, biventricle_mask_tensor: torch.Tensor, leftventricle_mask_tensor: torch.Tensor, window_size, iter_num=5000, init_deformation_param=None, weight_dict = None, if_registration=True, if_reset_affine=True):


        """
        biventricle_mask_tensor: the mask tensor of the biventricle 1*H*W*D
        leftventricle_mask_tensor: the mask tensor of the leftventricle 1*H*W*D
        """

        if weight_dict != None:
            self.weight_dict = weight_dict

    
        if init_deformation_param == None:
            self.deformation_param = nn.Parameter(torch.zeros_like(self.deformation_param), requires_grad=True)
        else:
            self.deformation_param = nn.Parameter(init_deformation_param.to(self.device), requires_grad=True)

        
        point_cloud_target_bi = torch.stack(torch.where(biventricle_mask_tensor[0] > 0.5),dim=1).float()
        # point_cloud_target_lv = torch.stack(torch.where(torch.abs(leftventricle_mask_tensor[0]-0.5)<0.1),dim=1).float()

        point_cloud_target_bi = 2*(point_cloud_target_bi)/torch.tensor(biventricle_mask_tensor.shape[1:4],device=self.device).float()-  1
        # point_cloud_target_lv = 2*(point_cloud_target_lv)/torch.tensor(leftventricle_mask_tensor.shape[1:4],device=self.device).float()-  1

        point_cloud_target_bi = point_cloud_target_bi[:,[2,1,0]]
        # point_cloud_target_lv = point_cloud_target_lv[:,[2,1,0]]

        point_cloud_target_bi = point_cloud_target_bi*window_size/200
        # point_cloud_target_lv = point_cloud_target_lv*window_size/200

        meshes_target_lv = cubify(leftventricle_mask_tensor, 0.2)

        meshes_target_lv = self.smoother.mesh_smooth(meshes_target_lv, num_iterations=2)

        meshes_target_lv = meshes_target_lv.update_padded((meshes_target_lv.verts_padded()*window_size/200).float())

        point_cloud_target_lv = sample_points_from_meshes(meshes_target_lv, 2*self.reg_sample_num)[0]

        if if_reset_affine:
            self.reset_affine_param()

        # global registration rigidly
        if if_registration:
            tt = 0
            while tt<5:
                if if_reset_affine:
                    self.reset_affine_param()
                R, s, T = self.global_register(point_cloud_target_bi, point_cloud_target_lv)
                print(torch.norm(R)*180/np.pi)
                if torch.norm(R)*180/np.pi < 30:
                    break
                print('regid registration trying: ', tt)
                tt += 1
            else:
                print('Rigid registration failed!')
            
            print(torch.norm(R)*180/np.pi)
        

        # graph harmonic fitting with the target: meshes_target_lv
        
        deformation_param, s, chamfer_loss= self.ghb_fitting(meshes_target_lv, iter_num= iter_num)

        fitted_meshes = self.forward()
            
        # fitted_meshes = self.smoother.mesh_smooth(fitted_meshes, num_iterations=5)

        # self.deformation_param = nn.Parameter(self.project_to_ghb_eig(fitted_meshes.verts_packed()))

        # fitted_meshes = self.forward()
        
        return fitted_meshes, self.deformation_param, self.R, self.s, self.T, chamfer_loss






if __name__ == "__main__":
    R, s, t = rigid_registration(np.random.randn(1000,3), np.random.randn(1000,3), sample_num=100, iter_num=2, update_scale=False)

    print(R, s, t)

    ghb = Graph_Harmonic_Deform(load_objs_as_meshes(["/home/yihao/data/ParaHearts/data/canonical_worldcoord.obj"]).to('cuda:0'))

    print(ghb().verts_packed().shape)

    print(ghb.project_to_ghb_eig(ghb().verts_packed()).shape)
        

def norm_vectors(input_: torch.Tensor):
    """
    input_: [N, 3]

    """
    norm = torch.norm(input_, dim=1, keepdim=True)
    input_ = input_ / norm
    return input_
    

# eigvec = eigvec[:,np.where(np.abs(eigval)>1e-3)[0]]
# eigval = eigval[np.where(np.abs(eigval)>1e-3)[0]]