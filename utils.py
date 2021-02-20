import numpy as np
from scipy.sparse.csgraph import shortest_path
from static.algorithms.symbol_FEM.symbol import Polynomial, PolynomialArray
from static.algorithms.symbol_FEM.interpolation import BasisFunctions


class FiniteDifference:
    def __init__(self, order, mesh, neighbor_distance=2):
        nn, nd = mesh.nodes.shape

        indices = BasisFunctions.integer2base(np.arange((order + 1) ** nd), base=order + 1, width=nd)
        indices = indices[np.sum(indices, axis=1) <= order, :][1:]

        array = [Polynomial(np.array([np.prod(1/np.maximum(1, ids))]), np.expand_dims(ids, axis=0)) for ids in indices]
        self.polys = PolynomialArray(array, shape=(array.__len__(), ))
        print("finite difference:", self.polys.sum(axis=0), sep='\n')

        graph = np.zeros(shape=[nn, nn], dtype=np.float)
        for spx in mesh.simplices:
            for i in range(1, nd + 1):
                for j in range(i):
                    graph[spx[i], spx[j]] = 1
                    graph[spx[j], spx[i]] = 1
        graph = shortest_path(graph)
        print("distance graph:", graph, sep='\n')
        self.neighbors = [[j for j, v in enumerate(graph[i, :]) if 1 <= v <= neighbor_distance] for i in range(nn)]

        # f0 + matrix@diff = fval
        self.matrix = [self.polys(mesh.nodes[ids, :] - mesh.nodes[[i], :]) for i, ids in enumerate(self.neighbors)]

        # diff = (matrix.T@matrix)^{-1}@matrix.T@(fval - f0)
        diff_tensor = []
        for i, mat in enumerate(self.matrix):
            diff = np.zeros(shape=[indices.__len__(), nn])
            inv = np.linalg.inv(mat.T@mat)@mat.T
            diff[:, self.neighbors[i]] += inv
            diff[:, i] -= np.sum(inv, axis=1)
            diff_tensor.append(diff)
        self.diff_tensor = np.stack(diff_tensor, axis=0)
