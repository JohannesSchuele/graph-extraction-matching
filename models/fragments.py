import os
import networkx as nx
import numpy as np
import numpy as np
import torch
from typing import Optional

from typing import NamedTuple, Sequence, Union
# datastructures
from pytorch3d.structures import Meshes
from torch import nn
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, TensorProperties
)
import copy
from pytorch3d.structures import Pointclouds

class FragmentMap(nn.Module):

    def __init__(self, renderer=None,  meshes=None, fragments_input = None,
                 pix_to_face=None, zbuf=None, bary_coords=None, dists=None, device='cpu') -> None:

        super().__init__()
        if isinstance(fragments_input, FragmentMap):
            self.pix_to_face=fragments_input.pix_to_face
            self.zbuf = fragments_input.zbuf
            self.bary_coords = fragments_input.bary_coords
            self.dists = fragments_input.dists
        elif isinstance(fragments_input, Fragments):
            self.pix_to_face=fragments_input.pix_to_face
            self.zbuf = fragments_input.zbuf
            self.bary_coords = fragments_input.bary_coords
            self.dists = fragments_input.dists
        else:
            self.pix_to_face=pix_to_face
            self.zbuf = zbuf
            self.bary_coords = bary_coords
            self.dists = dists

        self.renderer = renderer
        self.meshes= meshes
        self.device = device
        self.shape = self.pix_to_face.shape

    def add_new_fragments(self, new_fragments_input = None, new_pix_to_face=None,
                          new_zbuf=None, new_bary_coords=None, new_dists=None):
        if new_fragments_input is not None:
            self.pix_to_face = torch.cat((self.pix_to_face, new_fragments_input.pix_to_face), dim=2)
            self.zbuf = torch.cat((self.zbuf, new_fragments_input.zbuf), dim=2)
            self.bary_coords = torch.cat((self.bary_coords, new_fragments_input.bary_coords), dim=2)
            self.dists = torch.cat((self.dists, new_fragments_input.dists), dim=2)
        else:
            self.pix_to_face = torch.cat((self.pix_to_face, new_pix_to_face), dim=2)
            self.zbuf = torch.cat((self.zbuf, new_zbuf), dim=2)
            self.bary_coords = torch.cat((self.bary_coords, new_bary_coords), dim=2)
            self.dists = torch.cat((self.dists, new_dists), dim=2)

        self.shape = self.pix_to_face.shape

    def get_sub_points(self, indices):
        pix_to_face = self.pix_to_face[:, :, indices, :]
        zbuf = self.zbuf[:, :, indices, :]
        bary_coords = self.bary_coords[:, :, indices, :, :]
        dists = self.dists[:, :, indices, :]
        # return self.__class__(meshes=None, renderer=None,
        #                       pix_to_face= pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)
        return self.__class__(meshes=self.meshes, renderer=self.renderer,
                              pix_to_face=pix_to_face, zbuf=zbuf,
                              bary_coords=bary_coords, dists=dists,
                              device=self.device)

    def update_seen_fragments(self, indices, new_fragments_input = None, new_pix_to_face=None,
                          new_zbuf=None, new_bary_coords=None, new_dists=None):
        if new_fragments_input is not None:
            self.pix_to_face[:, :, indices, :] =  new_fragments_input.pix_to_face
            self.zbuf[:, :, indices, :] = new_fragments_input.zbuf
            self.bary_coords[:, :, indices, :, :] = new_fragments_input.bary_coords
            self.dists[:, :, indices, :]= new_fragments_input.dists
        else:
            self.pix_to_face[:, :, indices, :] = new_pix_to_face
            self.zbuf[:, :, indices, :] = new_zbuf
            self.bary_coords[:, :, indices, :, :]  = new_bary_coords
            self.dists[:, :, indices, :] = new_dists

    def update_point_cloud(self, updated_mesh):
        point_cloud = self.renderer.update_point_cloud_on_mesh(updated_mesh, fragments=self)
        return point_cloud






#ToDo: using the following class FragmentMap2 would need to have a update function simliar to this:
# The difference would be the direct inheritance from Fragments!
# import collections
#
# def updateTuple(NamedTuple,nameOfNamedTuple):
#     ## Convert namedtuple to an ordered dictionary, which can be updated
#     NamedTuple_asdict = NamedTuple._asdict()
#
#     ## Make changes to the required named attributes
#     NamedTuple_asdict['path']= 'www.google.com'
#
#     ## reconstruct the namedtuple using the updated ordered dictionary
#     updated_NamedTuple = collections.namedtuple(nameOfNamedTuple, NamedTuple_asdict.keys())(**NamedTuple_asdict)
#
#     return updated_NamedTuple
#
# Tuple = collections.namedtuple("Tuple", "path")
# NamedTuple = Tuple(path='www.yahoo.com')
# NamedTuple = updateTuple(NamedTuple, "Tuple")

#
#
# class FragmentMap2(Fragments):
#
#     def __new__(cls, renderer=None,  meshes=None, fragments_input = None,
#                  pix_to_face=None, zbuf=None, bary_coords=None, dists=None, device='cpu'):
#         if isinstance(fragments_input, Fragments):
#             self = super(FragmentMap2, cls).__new__(cls,  pix_to_face=fragments_input.pix_to_face, zbuf=fragments_input.zbuf,
#                              bary_coords= fragments_input.bary_coords, dists=fragments_input.dists)
#         else:
#             self = super(FragmentMap2, cls).__new__(cls,  pix_to_face=pix_to_face, zbuf=zbuf,
#                              bary_coords= bary_coords, dists=dists)
#
#         return self
#
#     def __init__(self, renderer=None,  meshes=None, fragments_input = None,
#                  pix_to_face=None, zbuf=None, bary_coords=None, dists=None, device='cpu') -> None:
#         self.renderer = renderer
#         self.meshes= meshes
#         self.device = device
#
#         self.shape = self.pix_to_face.shape
#
#     def add_new_fragments(self, new_fragments_input = None, new_pix_to_face=None,
#                           new_zbuf=None, new_bary_coords=None, new_dists=None):
#         if new_fragments_input is not None:
#             self.pix_to_face = torch.vstack((self.pix_to_face, new_fragments_input.pix_to_face))
#             self.zbuf = torch.vstack((self.zbuf, new_fragments_input.zbuf))
#             self.bary_coords = torch.vstack((self.bary_coords, new_fragments_input.bary_coords))
#             self.dists = torch.vstack((self.dists, new_fragments_input.dists))
#
#         else:
#             self.pix_to_face = torch.vstack((self.pix_to_face, new_pix_to_face))
#             self.zbuf = torch.vstack((self.zbuf, new_zbuf))
#             self.bary_coords = torch.vstack((self.bary_coords, new_bary_coords))
#             self.dists = torch.vstack((self.dists, new_dists))
#
#         self.shape = self.pix_to_face.shape
#
#
#
#     def get_sub_points(self, indices):
#         pix_to_face = self.pix_to_face[indices]
#         zbuf = self.zbuf[indices]
#         bary_coords = self.bary_coords[indices]
#         dists = self.dists[indices]
#         return self.__class__(meshes=None, renderer=None,
#                               pix_to_face= pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)
#
#
#     def update_seen_fragments(self, indices, new_fragments_input = None, new_pix_to_face=None,
#                           new_zbuf=None, new_bary_coords=None, new_dists=None):
#         if new_fragments_input is not None:
#             self.pix_to_face[indices] =  new_fragments_input.pix_to_face
#             self.zbuf[indices] = new_fragments_input.zbuf
#             self.bary_coords[indices] = new_fragments_input.bary_coords
#             self.dists[indices] = new_fragments_input.dists
#         else:
#             self.pix_to_face[indices] = new_pix_to_face
#             self.zbuf[indices] = new_zbuf
#             self.bary_coords[indices] = self.bary_coords, new_bary_coords
#             self.dists[indices] = new_dists
#
#
#     def update_point_cloud(self, updated_mesh):
#         point_cloud = self.renderer.update_point_cloud_on_mesh(updated_mesh, fragments=self)
#         return point_cloud
#
#












