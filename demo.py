import numpy as np
from skimage import measure

from ddm import train
from mesh import IsotropicMesh
from utils import FiniteDifference

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cassini_oval_region(num=16):
    """(x^2 + y^2)^2 - 2(x^2 - y^2) = 3"""

    # generate dense points in bound
    phi = np.linspace(0, 2 * np.pi, num=10000, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    dense_boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in bound
    phi = np.linspace(0, 2 * np.pi, num=num, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in interal region
    gap = np.min(np.linalg.norm(boundary_points[:-1] - boundary_points[1:], axis=1))
    xv, yv = np.meshgrid(np.arange(-2, 2, gap), np.arange(-1.5, 1.5, gap))
    unsafe_inner_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    masks = measure.points_in_poly(points=unsafe_inner_points, verts=boundary_points)
    unsafe_inner_points = unsafe_inner_points[[i for i, mask in enumerate(masks) if mask]]
    distance = np.linalg.norm(np.expand_dims(unsafe_inner_points, axis=1) - np.expand_dims(dense_boundary_points, axis=0), axis=-1)
    inner_points = unsafe_inner_points[[i for i, dis in enumerate(np.min(distance, axis=1)) if dis > 0.5 * gap]]

    return boundary_points, inner_points


def unit_test(num=16, num_refine=0):
    # ----------------------------- generate non-uniform mesh -----------------------------
    boundary_nodes, inner_nodes = cassini_oval_region(num=num)
    nodes = train(boundary_nodes, inner_nodes, eps=1e-2, infimum=1e-1, n_store=10)

    mesh = IsotropicMesh(nodes)
    boundary_indices = [i for i, flag in enumerate(mesh.masks) if flag]
    inner_indices = [i for i, flag in enumerate(mesh.masks) if not flag]

    # ----------------------------- finite difference -----------------------------
    fd = FiniteDifference(order=2, mesh=mesh, neighbor_distance=1)

    f00 = np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f01 = -np.cos(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f02 = -np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f10 = -np.sin(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f11 = np.sin(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f20 = -np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])

    diff = np.einsum("nps,s->np", fd.diff_tensor, f00)
    print('error fy:', np.max(np.abs(diff[inner_indices, 0] - f01[inner_indices])))
    print('error fyy:', np.max(np.abs(diff[inner_indices, 1] - f02[inner_indices])))
    print('error fx:', np.max(np.abs(diff[inner_indices, 2] - f10[inner_indices])))
    print('error fxy:', np.max(np.abs(diff[inner_indices, 3] - f11[inner_indices])))
    print('error fxx:', np.max(np.abs(diff[inner_indices, 4] - f20[inner_indices])))

    # ----------------------------- generate equations -----------------------------
    laplace = np.sum(fd.diff_tensor, axis=1) + np.eye(mesh.nodes.__len__())
    matrix = laplace[inner_indices, :][:, inner_indices]
    rhs = (f20 + f02 + f11 + f10 + f01 + f00)[inner_indices] - laplace[inner_indices, :][:, boundary_indices]@f00[boundary_indices]
    numerical_f00 = np.zeros_like(f00)
    numerical_f00[boundary_indices] = f00[boundary_indices]
    numerical_f00[inner_indices] = np.linalg.solve(matrix, rhs)
    print("L0 error:", np.max(np.abs(numerical_f00 - f00)))

    # ----------------------------- draw in canvas -----------------------------
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("numerical solution")
    ax.plot_trisurf(mesh.nodes[:, 0], mesh.nodes[:, 1], numerical_f00[:], alpha=0.5)

    ax = fig.add_subplot(122, projection='3d')
    ax.set_title("error")
    ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], np.zeros_like(mesh.nodes[:, 0]))
    for i, (x, y) in enumerate(mesh.nodes):
        ax.text(x, y, 0, str(i))
    ax.plot_trisurf(mesh.nodes[:, 0], mesh.nodes[:, 1], numerical_f00 - f00, alpha=0.5)

    plt.show()


if __name__ == "__main__":
    unit_test(num=16, num_refine=0)

    # ----------------------------- generate non-uniform mesh -----------------------------
    boundary_nodes, inner_nodes = cassini_oval_region(num=32)
    nodes = train(boundary_nodes, inner_nodes, eps=1e-2, infimum=1e-1, n_store=10)
    mesh = IsotropicMesh(nodes)
    boundary_indices = [i for i, flag in enumerate(mesh.masks) if flag]
    inner_indices = [i for i, flag in enumerate(mesh.masks) if not flag]

    # ----------------------------- finite difference -----------------------------
    fd = FiniteDifference(order=3, mesh=mesh, neighbor_distance=2)

    f00 = np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f01 = -np.cos(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f02 = -np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f03 = np.cos(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f10 = -np.sin(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f11 = np.sin(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f12 = np.sin(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f20 = -np.cos(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])
    f21 = np.cos(mesh.nodes[:, 0]) * np.sin(mesh.nodes[:, 1])
    f30 = np.sin(mesh.nodes[:, 0]) * np.cos(mesh.nodes[:, 1])

    diff = np.einsum("nps,s->np", fd.diff_tensor, f00)
    print('error fy:', np.max(np.abs(diff[inner_indices, 0] - f01[inner_indices])))
    print('error fyy:', np.max(np.abs(diff[inner_indices, 1] - f02[inner_indices])))
    print('error fyyy:', np.max(np.abs(diff[inner_indices, 2] - f03[inner_indices])))
    print('error fx:', np.max(np.abs(diff[inner_indices, 3] - f10[inner_indices])))
    print('error fxy:', np.max(np.abs(diff[inner_indices, 4] - f11[inner_indices])))
    print('error fxyy:', np.max(np.abs(diff[inner_indices, 5] - f12[inner_indices])))
    print('error fxx:', np.max(np.abs(diff[inner_indices, 6] - f20[inner_indices])))
    print('error fxxy:', np.max(np.abs(diff[inner_indices, 7] - f21[inner_indices])))
    print('error fxxx:', np.max(np.abs(diff[inner_indices, 8] - f30[inner_indices])))
