
import torch
import torch
from pytorch3d.ops import cot_laplacian

def mesh_consistency_edge_loss(meshes):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()


    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    target_length = torch.mean(torch.norm((v0 - v1), dim=1)) #ToDo: test which dim!

    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N







def mesh_normal_consistency_between_meshes(meshes, mesh_target, use_face_normals= True):
    r"""
    Computes the normal consistency of each mesh in meshes.
    We compute the normal consistency for each pair of neighboring faces.
    If e = (v0, v1) is the connecting edge of two neighboring faces f0 and f1,
    then the normal consistency between f0 and f1

    .. code-block:: python

                    a
                    /\
                   /  \
                  / f0 \
                 /      \
            v0  /____e___\ v1
                \        /
                 \      /
                  \ f1 /
                   \  /
                    \/
                    b

    The normal consistency is

    .. code-block:: python

        nc(f0, f1) = 1 - cos(n0, n1)

        where cos(n0, n1) = n0^n1 / ||n0|| / ||n1|| is the cosine of the angle
        between the normals n0 and n1, and

        n0 = (v1 - v0) x (a - v0)
        n1 = - (v1 - v0) x (b - v0) = (b - v0) x (v1 - v0)

    This means that if nc(f0, f1) = 0 then n0 and n1 point to the same
    direction, while if nc(f0, f1) = 2 then n0 and n1 point opposite direction.

    .. note::
        For well-constructed meshes the assumption that only two faces share an
        edge is true. This assumption could make the implementation easier and faster.
        This implementation does not follow this assumption. All the faces sharing e,
        which can be any in number, are discovered.

    Args:
        meshes: Meshes object with a batch of meshes.

    Returns:
        loss: Average normal consistency across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    if use_face_normals is True:
        loss = 1 - torch.cosine_similarity(meshes.faces_normals_packed(), mesh_target.faces_normals_packed(), dim=1)
    else:
        loss = 1 - torch.cosine_similarity(meshes.verts_normals_packed(), mesh_target.verts_normals_packed(), dim=1)

    return loss.mean()



def mesh_laplacian_consistency_smoothing(meshes, mesh_target, method: str = "uniform"):

    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].



    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    def calc_laplacian(meshes, method):
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        # We don't want to backprop through the computation of the Laplacian;
        # just treat it as a magic constant matrix that is used to transform
        # verts into normals
        with torch.no_grad():
            if method == "uniform":
                L = meshes.laplacian_packed()
            elif method in ["cot"]:
                L, inv_areas = cot_laplacian(verts_packed, faces_packed)
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                raise ValueError("Method should be one of {uniform, cot}")

        if method == "uniform":
            lap = L.mm(verts_packed)
        elif method == "cot":
            lap = L.mm(verts_packed) * norm_w - verts_packed
        else:
            raise ValueError("Method should be one of {uniform, cot}")
        return lap

    lap_meshes = calc_laplacian(meshes=meshes, method=method)
    lap_mesh_target = calc_laplacian(meshes=mesh_target, method=method)

    loss = torch.norm((lap_meshes-lap_mesh_target), dim=1)

    return loss.mean()
