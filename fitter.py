from mesh_function.Graph_Harmonic_Fitting import Graph_Harmonic_Deform
from typing import List, Union, Dict
from losses.meshloss import Mesh_loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
import numpy as np
import torch
import os
from pytorch3d.io import save_obj
import pickle
import open3d as o3d
from mesh_function.Graph_Harmonic_Fitting import norm_vectors


class AAA_Registration(object):
    def __init__(self, pcds: List, AAA_OuterMesh, loss_weighting: Dict, root,
                 redo_alignment=False, redo_ghd=False,
                 num_GHD=4 ** 2,
                 use_cpu=False,
                 meta="debug",
                 visualize_alignment=False):
        # normalize
        self.meta = meta
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not use_cpu and not visualize_alignment else "cpu")
        # additional viz options
        self.visualize_alignment = visualize_alignment

        AAA_OuterMesh = safe_load_mesh(AAA_OuterMesh)
        radius = torch.std(torch.norm(AAA_OuterMesh.verts_packed(), dim=1))
        center = torch.mean(AAA_OuterMesh.verts_packed(), dim=0, keepdim=True)
        new_verts = (AAA_OuterMesh.verts_packed() - center) / radius
        AAA_OuterMesh = AAA_OuterMesh.update_padded(new_verts.unsqueeze(0))
        pcds = [(pcd - torch.mean(pcd[0, ...], dim=0, keepdim=True)) / radius for pcd in pcds]
        self.pcds = pcds
        self.losser = Mesh_loss(AAA_OuterMesh.to(self.device))
        self.loss_weighting = loss_weighting
        self.AAA_trimesh = trimesh.Trimesh(vertices=AAA_OuterMesh.verts_packed().cpu().numpy(),
                                           faces=AAA_OuterMesh.faces_packed().cpu().numpy())
        self.graph_fitter = Graph_Harmonic_Deform(AAA_OuterMesh, num_Basis=num_GHD)
        self.graph_fitter.to(self.device)
        self.AAA_OuterMesh = AAA_OuterMesh

        # align pcd -> CT mesh
        self.root = root
        alignment_chk_path = os.path.join(root, self.meta, "alignment_chk.pkl")
        if not os.path.exists(alignment_chk_path) or redo_alignment:
            self.aligned_pcds_ED, self.aligned_pcds_ES, self.pca_Mesh, self.aligned_pca_echo = self.align_pcds()
            chk = {"aligned_pcds_ED": self.aligned_pcds_ED,
                   "aligned_pcds_ES": self.aligned_pcds_ES,
                   "pca_Mesh": self.pca_Mesh,
                   "aligned_pca_echo": self.aligned_pca_echo}
            with open(alignment_chk_path, 'wb') as f:
                pickle.dump(chk, f)
        else:
            with open(alignment_chk_path, 'rb') as f:
                chk = pickle.load(f)
                for key in chk.keys():
                    setattr(self, key, chk[key])

        # normal estimation
        self.aligned_pcd_ED_norm, self.aligned_pcd_ES_norm = self.normal_estimation(visualize=False)

        # GHD fitting: CT mesh -> aligned pcd
        self.GHD_coefficients = []
        self.num_GHD = num_GHD
        ghd_chk_path = os.path.join(root, self.meta, "ghd_chk.pkl")
        if not os.path.exists(ghd_chk_path) or redo_ghd:
            self.ghd_coefficients_all, self.fitted_verts_all_frames = self.fit_GHD()
            chk = {"ghd_coefficients_all": self.ghd_coefficients_all,
                   "fitted_verts_all_frames": self.fitted_verts_all_frames}
            with open(ghd_chk_path, 'wb') as f:
                pickle.dump(chk, f)
        else:
            with open(ghd_chk_path, 'rb') as f:
                chk = pickle.load(f)
                for key in chk.keys():
                    setattr(self, key, chk[key])

    def reset_RT(self):
        self.R = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.s = torch.tensor([1.], device=self.device, requires_grad=False).unsqueeze(0)
        self.T = nn.Parameter(torch.zeros(1, 3, device=self.device))

    # def reset_GHD_RT(self):
    #     # self.R_ = nn.Parameter(torch.zeros(1, 3, device=self.device))
    #     """
    #     canonical shpae does not have rigid transform freedom.
    #     """
    #     self.R_ = torch.zeros(1, 3, device=self.device, requires_grad=False)
    #     self.s_ = torch.tensor([1.], device=self.device, requires_grad=False).unsqueeze(0)
    #     self.T_ = torch.zeros(1, 3, device=self.device, requires_grad=False)
    def reset_GHD_coefficients(self):
        self.ghd_coefficients = nn.Parameter(torch.zeros((self.num_GHD, 3), dtype=torch.float32, device=self.device),
                                             requires_grad=True)
        self.ghd_coefficients_radial = nn.Parameter(
            torch.zeros((self.num_GHD, 1), dtype=torch.float32, device=self.device),
            requires_grad=True)
        self.mapping_coefficients = nn.Parameter(torch.ones((1, 3), dtype=torch.float32, device=self.device),
                                                 requires_grad=True)

    def align_pcds(self):
        Mesh = getattr(self.graph_fitter, "base_shape").to(self.device)
        pca = decomposition.PCA(n_components=2)
        pcd_mesh = sample_points_from_meshes(Mesh, 5000)
        pca.fit(pcd_mesh.squeeze(0).detach().cpu().numpy())
        pca_Mesh = torch.Tensor(pca.components_).to(self.device)

        i = 0
        aligned_pcds_ED = []
        aligned_pcds_ES = []
        pca_echo = []
        aligned_pca_echo = []
        self.pca_direction_map = [1, 1, -1]

        for pcd_EDES, pca_direction_map_ in zip(self.pcds, self.pca_direction_map):
            pcd_EDES = pcd_EDES.to(self.device)
            pcd_ED = pcd_EDES[0, ...]
            pcd_ES = pcd_EDES[1, ...]

            # pca
            pca = decomposition.PCA(n_components=2)
            pca.fit(pcd_ED.detach().cpu().numpy())
            W = pca.components_
            pca_pcd = torch.Tensor(pca.components_).to(self.device)
            pca_pcd[0, :] = pca_pcd[0, :] * pca_direction_map_
            pca_echo.append(pca_pcd)

            if self.visualize_alignment:
                p = pv.Plotter()
                arrow1 = pv.Arrow(np.array([0, 0, 0]), pca_pcd[0, :].numpy(), scale=5)
                arrow2 = pv.Arrow(np.array([0, 0, 0]), pca_pcd[1, :].numpy(), scale=5)
                # arrow3 = pv.Arrow(np.array([0, 0, 0]), W[2, :], scale=5)
                # Add the arrow to the plotter
                p.add_mesh(arrow1, color='red')
                p.add_mesh(arrow2, color='blue')
                # p.add_mesh(arrow3, color='yellow')
                p.add_points(pcd_ED.detach().cpu().numpy(),
                             render_points_as_spheres=True, color='blue')
                Mesh_centroid = self.AAA_OuterMesh.verts_packed().detach().cpu().mean(dim=0).numpy()
                arrow1_Mesh = pv.Arrow(Mesh_centroid, pca_Mesh[0, :].cpu().numpy(), scale=5)
                arrow2_Mesh = pv.Arrow(Mesh_centroid, pca_Mesh[1, :].cpu().numpy(), scale=5)
                p.add_mesh(self.AAA_trimesh, color='green', opacity=0.5)
                p.add_mesh(arrow1_Mesh, color='black')
                p.add_mesh(arrow2_Mesh, color='white')
                p.show()

        for pcd_EDES, pca_pcd in zip(self.pcds, pca_echo):
            pcd_EDES = pcd_EDES.to(self.device)
            pcd_ED = pcd_EDES[0, ...]
            pcd_ES = pcd_EDES[1, ...]

            self.reset_RT()
            # self.T = nn.Parameter((torch.mean(Mesh.verts_packed(), dim=0) - torch.mean(pcd_ED, dim=0)).to(self.device))
            optimizer = torch.optim.AdamW([self.R, self.T], lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
            pca_loss_module = torch.nn.MSELoss()
            for epoch in range(1000):
                R_matrix = axis_angle_to_matrix(self.R)
                pcd_RT = (pcd_ED.unsqueeze(0) @ R_matrix.transpose(-1, -2) * self.s + self.T).float()
                pca_pcd_RT = (pca_pcd.unsqueeze(0) @ R_matrix.transpose(-1, -2) * self.s).float().squeeze(0)
                pcd_mesh = sample_points_from_meshes(Mesh, 10000)
                pcd_mesh = point_filter(pcd_mesh.squeeze(0), pcd_RT.squeeze(0), pca_pcd_RT[0, :].view(1, 3)).unsqueeze(
                    0)
                chamfer_loss, _ = chamfer_distance(pcd_mesh, pcd_RT)
                # pca_loss = 1 * pca_loss_module(torch.abs(torch.sum(pca_pcd_RT * pca_Mesh, dim=1)),
                #                                torch.Tensor([1, 1]).to(self.device))
                pca_loss = 1 * pca_loss_module(torch.sum(pca_pcd_RT * pca_Mesh, dim=1),
                                               torch.Tensor([1, 1]).to(self.device))
                loss = pca_loss + 1 * chamfer_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                print("Epoch {}, chamfer loss {}, pca loss {}".format(epoch, chamfer_loss.item(), pca_loss.item()))
            pcd_ED = (pcd_ED.unsqueeze(0) @ R_matrix.transpose(-1, -2) * self.s + self.T).float().squeeze(
                0).detach().cpu()
            pcd_ES = (pcd_ES.unsqueeze(0) @ R_matrix.transpose(-1, -2) * self.s + self.T).float().squeeze(
                0).detach().cpu()

            aligned_pcds_ED.append(pcd_ED)
            aligned_pcds_ES.append(pcd_ES)
            aligned_pca_echo.append(
                (pca_pcd.unsqueeze(0) @ R_matrix.transpose(-1, -2) * self.s).float().squeeze(0).detach().cpu())
            plot_alignment(Mesh, pcd_ED, pcd_ES,
                           fig_path=os.path.join(self.root, self.meta, "alignment_" + str(i)),
                           show=True)
            i += 1
        return aligned_pcds_ED, aligned_pcds_ES, pca_Mesh, aligned_pca_echo

    def fit_GHD(self):
        i = 0
        ghd_coefficients_all = []
        fitted_verts_all_frames = []
        for aligned_pcds_ED, aligned_pcds_ES, aligned_pca_echo, aligned_pcd_ED_norm, aligned_pcd_ES_norm in zip(
                self.aligned_pcds_ED, self.aligned_pcds_ES,
                self.aligned_pca_echo,
                self.aligned_pcd_ED_norm,
                self.aligned_pcd_ES_norm):
            i += 1
            ghd_coefficients_frame = []
            fitted_verts_sgl_frame = {'ED': None, 'ES': None}
            for label_, echo_pcd, aligned_pcd_norm in zip(["ED", "ES"], [aligned_pcds_ED, aligned_pcds_ES], [aligned_pcd_ED_norm, aligned_pcd_ES_norm]):
                log_path = os.path.join(self.root, self.meta, "frame_" + str(i) + "_" + label_)
                self.graph_fitter.reset_affine_param()
                self.reset_GHD_coefficients()
                optimizer = torch.optim.AdamW(
                    [self.ghd_coefficients, self.ghd_coefficients_radial,
                     self.graph_fitter.R, self.graph_fitter.s, self.graph_fitter.T],
                    lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.75)
                total_epoch = 25000
                for epoch in range(total_epoch):
                    # warped_Mesh = self.graph_fitter.forward(self.ghd_coefficients)
                    # warped_Mesh = self.graph_fitter.forward_radial(self.ghd_coefficients,
                    #                                                self.ghd_coefficients_radial,
                    #                                                # self.pca_Mesh.to(self.device)[0, :],
                    #                                                aligned_pca_echo[0, :],
                    #                                                centroid=torch.mean(echo_pcd, dim=0).to(self.device))
                    warped_Mesh, rigid_static_Mesh = self.graph_fitter.forward_normal(self.ghd_coefficients,
                                                                                      self.ghd_coefficients_radial,
                                                                                      self.mapping_coefficients
                                                                                      )
                    loss_dict = self.losser.forward(warped_Mesh, echo_pcd, aligned_pca_echo.to(self.device),
                                                    self.loss_weighting,
                                                    rigid_static_Mesh,
                                                    aligned_pcd_norm.to(self.device))
                    total_loss = torch.zeros(1, device=self.device)
                    log_dict = {}
                    log_dict['epoch'] = epoch
                    for term, loss in loss_dict.items():
                        if term == "loss_rigid":
                            total_loss += loss * self.loss_weighting[term] * (1 - epoch / total_epoch + 0.05)
                        # elif term == "axis_reg_loss" or term == "bound_reg_loss":
                        #     total_loss += loss * self.loss_weighting[term] * (epoch / total_epoch * 0.5)
                        else:
                            total_loss += loss * self.loss_weighting[term]
                        log_dict[term] = (loss * self.loss_weighting[term]).cpu().item()
                    log_dict_printer(log_dict)

                    # gradient descent
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    viz_fitting_static(epoch, log_path, warped_Mesh, echo_pcd)
                ghd_coefficients_frame.append({"ghd_coefficients": self.ghd_coefficients.detach().cpu(),
                                               "R": self.graph_fitter.R.detach().cpu(),
                                               "s": self.graph_fitter.s.detach().cpu(),
                                               "T": self.graph_fitter.T.detach().cpu()})
                fitted_verts_sgl_frame[label_] = warped_Mesh.verts_packed().detach().cpu()

            ghd_coefficients_all.append(ghd_coefficients_frame)
            fitted_verts_all_frames.append(fitted_verts_sgl_frame)
        return ghd_coefficients_all, fitted_verts_all_frames

    def normal_estimation(self, visualize=False):
        aligned_pcd_ED_norm = []
        aligned_pcd_ES_norm = []
        for aligned_pcd_ED, aligned_pcd_ES, aligned_pca_echo in zip(self.aligned_pcds_ED, self.aligned_pcds_ES,
                                                                    self.aligned_pca_echo):
            for i, aligned_pcd in enumerate([aligned_pcd_ED, aligned_pcd_ES]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(aligned_pcd.numpy())

                # # debug: pca vector cannot go through pcd!!!
                # # if this happens, we can use mesh norm as dot target.
                # p = pv.Plotter()
                # p.add_points(aligned_pcd.numpy())
                # arrow1 = pv.Arrow(np.array(aligned_pcd.mean(dim=0).numpy()), aligned_pca_echo[0, :].cpu().numpy(), scale=5)
                # p.add_mesh(arrow1)
                # p.show()

                # Estimate normals if needed
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
                )
                normals = np.asarray(pcd.normals)

                flipped_normals = self.flip_normals(aligned_pcd, normals, aligned_pca_echo)
                pcd.normals = o3d.utility.Vector3dVector(flipped_normals.numpy())
                if visualize:
                    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
                if i == 0:
                    aligned_pcd_ED_norm.append(torch.Tensor(pcd.normals))
                else:
                    aligned_pcd_ES_norm.append(torch.Tensor(pcd.normals))

        return aligned_pcd_ED_norm, aligned_pcd_ES_norm

    @staticmethod
    def flip_normals(pcd, pcd_normals, pca_vector):
        """
        C = A + t[a, b, c]
        t = a(xB-xA)+b(yB-yA)+c(zC-zA)/(a^2+b^2+c^2)
        """
        if not isinstance(pcd_normals, torch.Tensor):
            pcd_normals = torch.Tensor(pcd_normals)
        pca_vector = pca_vector[0, :].view(1, 3).to(pcd.device)
        A = pcd.detach().cpu().mean(dim=0, keepdim=True)
        AB = pcd - A
        t = torch.sum(pca_vector * AB, dim=1, keepdim=True) / torch.norm(pca_vector, dim=1, keepdim=True)
        C = A + t * pca_vector
        signs = torch.sign(torch.sum(norm_vectors(pcd - C) * norm_vectors(pcd_normals), dim=1, keepdim=True))
        pcd_normals = pcd_normals * signs
        return pcd_normals


def log_dict_printer(log_dict: dict):
    print({key: '{:.5f}'.format(value) for key, value in log_dict.items()})


def viz_fitting_static(epoch: int, log_path, warped_mesh: Meshes, target_pcd: torch.Tensor) -> None:
    if epoch % 1000 == 0 or epoch == 19999:
        lazy_plot_meshes(warped_mesh, target_pcd, os.path.join(log_path, str(epoch).zfill(5) + '.jpg'))
        save_obj(os.path.join(log_path, str(epoch).zfill(5) + '.obj'), verts=warped_mesh.verts_packed(),
                 faces=warped_mesh.faces_packed())


def lazy_plot_meshes(canonical_mesh, target_pcd, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # x, y, z = point_cloud_sampling.clone().detach().cpu().squeeze().unbind(1)
    # ax.scatter3D(z, y, x, c=z, cmap='Greens', s=0.5)
    T = canonical_mesh.faces_packed().detach().cpu().numpy()
    vertices = canonical_mesh.verts_packed().detach().cpu().numpy()

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=T, edgecolor=[[0, 0, 0]], linewidth=0.01,
                    alpha=0.5, color='yellowgreen')
    target_pcd = target_pcd.detach().cpu().numpy()
    ax.scatter(target_pcd[:, 0], target_pcd[:, 1], target_pcd[:, 2], color='black', s=0.5)
    ax.view_init(elev=5, azim=45)
    if not save_path.endswith('jpg'):
        save_path += '.jpg'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def plot_alignment(Mesh, pcd_ED, pcd_ES,
                   fig_path, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T = Mesh.faces_packed().detach().cpu().numpy()
    vertices = Mesh.verts_packed().detach().cpu().numpy()

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=T, edgecolor=[[0, 0, 0]],
                    linewidth=0.01,
                    alpha=0.2, color='yellowgreen')
    ax.scatter(pcd_ED.numpy()[:, 0], pcd_ED.numpy()[:, 1], pcd_ED.numpy()[:, 2], color='blue', s=0.5)
    ax.scatter(pcd_ES.numpy()[:, 0], pcd_ES.numpy()[:, 1], pcd_ES.numpy()[:, 2], color='red', s=0.5)
    ax.view_init(elev=5, azim=45)
    # fig_path = os.path.join(self.root, self.meta, "alignment_" + str(i))
    if fig_path is not None:
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))
        plt.savefig(fig_path, dpi=500)
    if show:
        plt.show()
    plt.close(fig)


def point_filter(pcd1, pcd2, vector):
    """

    pcd: [N, 3]
    vector: [1, 3]
    pcd1: pcd to filter
    pcd2: pcd for measure
    """
    project1 = torch.matmul(pcd1, vector.transpose(0, 1))
    project2 = torch.matmul(pcd2, vector.transpose(0, 1))
    max_ = torch.max(project2)
    min_ = torch.min(project2)
    indices = torch.where((project1 >= min_) & (project1 <= max_))[0]
    return pcd1[indices, :]


def safe_load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    verts = torch.Tensor(np.asarray(mesh.vertices))
    faces = torch.Tensor(np.asarray(mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh


"""
sudo docker run --name wenhao_pvae1 --shm-size=2g --gpus '"device=0"' --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v '/media/yaplab/"HDD Storage"/wenhao/aaareg:/workspace' -it --rm ghb_ia:v0 bash
pip install einops
sudo mkdir -p '/media/yaplab/"HDD Storage"/wenhao/aaareg'
sudo chmod -R 777 '/media/yaplab/"HDD Storage"/wenhao'
"""
