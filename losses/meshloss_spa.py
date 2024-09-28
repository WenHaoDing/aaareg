import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from .meshloss import Mesh_loss
from utils_oa.fn_opening_alignment import opening_alignment_pca
from utils_oa.fn_surface_alignment import surface_alignment_pca
import torch
from pytorch3d.structures import Meshes
import numpy as np
from .meshloss_oa import Mesh_loss_opening_alignment


class Mesh_loss_surface_partition_alignment(Mesh_loss_opening_alignment):
    def __init__(self, args, spa_class_canonical: surface_alignment_pca, spa_class_target: surface_alignment_pca,
                 release_idx: list = None):
        # return_surface_partition_Meshes_static
        spa_sample_ratio = getattr(args, 'spa_sample_ratio', [0.5, 0.5])  # if not given then 50 50
        self.spa_sample_num = [int(args.sample_num * ratio) for ratio in spa_sample_ratio]
        self.target_surface_partition = spa_class_target.return_surface_partition_Meshes_static()
        self.release_idx = release_idx if release_idx is not None else []  # surface partitions to release
        super(Mesh_loss_surface_partition_alignment, self).__init__(args, spa_class_canonical, spa_class_target)

    def forward_surface_partition_alignment(self, warped_Mesh, warped_openings, warped_surface_partitions,
                                            loss_weighting: dict, B=1):
        loss_dict = self.forward_opening_alignment(warped_Mesh, warped_openings, loss_weighting, B)
        loss_p_list = []
        loss_n_list = []
        if ('loss_surface_partition_p' in loss_weighting) or ('loss_surface_partition_n' in loss_weighting):
            for idx in [face for face in range(len(warped_surface_partitions)) if face not in self.release_idx]:
                pcd_wsp, nor_wsp = sample_points_from_meshes(warped_surface_partitions[idx], self.spa_sample_num[idx],
                                                             return_normals=True)
                pcd_tsp, nor_tsp = sample_points_from_meshes(self.target_surface_partition[idx],
                                                             self.spa_sample_num[idx], return_normals=True)
                loss_p, loss_n = chamfer_distance(pcd_wsp, pcd_tsp, x_normals=nor_wsp, y_normals=nor_tsp)
                loss_p_list.append(loss_p)
                loss_n_list.append(loss_n)
            if 'loss_surface_partition_p' in loss_weighting:
                loss_dict['loss_surface_partition_p'] = loss_p_list
            if 'loss_surface_partition_n' in loss_weighting:
                loss_dict['loss_surface_partition_n'] = loss_n_list
        return loss_dict
