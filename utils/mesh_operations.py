import os
import torch
from pytorch3d.utils import ico_sphere
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes


def pool_mesh_to_dim_of_original_mesh(subdivided_mesh, original_mesh):
    idx_verts = original_mesh.num_verts_per_mesh()
    idx_faces = original_mesh.num_faces_per_mesh()
    verts = subdivided_mesh.verts_packed()[range(idx_verts)]
    #verts /= verts.norm(p=2, dim=1, keepdim=True)
    faces = subdivided_mesh.faces_packed()[range(idx_faces)]
    return Meshes(verts=[verts], faces=[faces])


def subdivideMesh(mesh, iter=1):
    subdivide = SubdivideMeshes()
    sub_divided_mesh = mesh
    for i in range(iter):
        sub_divided_mesh = subdivide(sub_divided_mesh)
    return sub_divided_mesh


if __name__ == "__main__":

    src_mesh = ico_sphere(2, device)
    divided_mesh = subdivideMesh(mesh=src_mesh, iter=3)
    print('number of faces of subdivided mesh: ', divided_mesh.num_faces_per_mesh())
    pooled_mesh = pool_mesh_to_dim_of_original_mesh(subdivided_mesh=divided_mesh, original_mesh=src_mesh)
    print('Boolean to check if mesh vertices correspond to each other: ', src_mesh.verts_packed()==pooled_mesh.verts_packed())


