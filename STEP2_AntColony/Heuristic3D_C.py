"""
将三维粒子坐标转化为特定的目标函数矩阵，调用C++加速
"""

import enum
import ctypes
import numpy as np
import scipy.spatial
from itertools import product
from tkinter import _flatten

import Heuristic3D


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    DistanceGraph = '最短位移'
    VoronoiGraph = '泰森最相似'
    RelaxGraph = '泰森松弛距离'
    OldRelaxGraph = '最紧邻松弛距离'
    MixedGraph = '混合判据'


# 指定用到的动态链接库，以及库内函数的参数表
heuristic_dll = ctypes.cdll.LoadLibrary(r'Heuristic3D.dll')
heuristic_dll.CalRidge.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int), ctypes.c_int)
heuristic_dll.VoronoiPolar.argtypes = (ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
heuristic_dll.VoronoiRelation.argtypes = (ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int)
heuristic_dll.CalNeighbor.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_int)
heuristic_dll.RelaxMethod.argtypes = (ctypes.POINTER(ctypes.c_double),
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
heuristic_dll.OldCalNeighbor.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)
heuristic_dll.OldRelaxMethod.argtypes = (ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))


class DistanceGraph(Heuristic3D.DistanceGraph):
    """用三维最短距离作为目标函数，提供统一接口，并未加速"""
    pass


class VoronoiGraph(Heuristic3D.VoronoiGraph):
    """用三维Voronoi多面体最相似作为目标函数（Nie2021 DPF-PTV），调用C++加速
    Reference: A hybrid 3D particle matching algorithm based on ant colony optimization
    """
    def _cal_graph(self):
        # 八个边角点加入，用于防止边角处Voronoi多边形不封闭
        pos_min = np.minimum(np.min(self._src_particles, axis=0), np.min(self._dest_particles, axis=0)) - 1e-5
        pos_max = np.maximum(np.max(self._src_particles, axis=0), np.max(self._dest_particles, axis=0)) + 1e-5
        margin_particle = np.array(list(product(*zip(pos_min, pos_max))))
        src_particle = np.vstack((self._src_particles, margin_particle))
        dest_particle = np.vstack((self._dest_particles, margin_particle))

        # 调用库函数直接求解Voronoi分割
        src_voronoi = scipy.spatial.Voronoi(src_particle)
        dest_voronoi = scipy.spatial.Voronoi(dest_particle)

        # 初始缓冲区大小为50，若不够按两倍扩容
        # 调用C++，根据边界面对应的两侧粒子索引ridge_points，将边界面添加至各粒子的Voronoi多面体边界面索引数组particle_ridges中，即particle_ridges为ridge_points交换index与value的数组
        max_ridges = 50
        ret_flag = -1
        while ret_flag < 0:
            src_particle_ridges = (ctypes.c_int * (max_ridges * len(src_particle)))()
            src_particle_ridges_len = (ctypes.c_int * len(src_particle))()
            ret_flag = heuristic_dll.CalRidge(src_particle_ridges, src_particle_ridges_len, max_ridges,
                                              src_voronoi.ridge_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                              len(src_voronoi.ridge_points))

            dest_particle_ridges = (ctypes.c_int * (max_ridges * len(dest_particle)))()
            dest_particle_ridges_len = (ctypes.c_int * len(dest_particle))()
            ret_flag += heuristic_dll.CalRidge(dest_particle_ridges, dest_particle_ridges_len, max_ridges,
                                               dest_voronoi.ridge_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                               len(dest_voronoi.ridge_points))
            max_ridges *= 2
        max_ridges //= 2

        # 先行计算单位方向矢量（自变量的离散值），以中心粒子为原点，发射的该向量组与其对应的Voronoi多面体的交点为待求点（可参照Nie2021 DPF-PTV论文）
        num = 2 * self._STEP * self._STEP
        theta = np.arange(0, 2 * np.pi, np.pi / self._STEP)
        fai = np.arange(0.5 * np.pi / self._STEP, np.pi, np.pi / self._STEP)
        sin_theta = np.tile(np.sin(theta), self._STEP)
        cos_theta = np.tile(np.cos(theta), self._STEP)
        sin_fai = np.tile(np.sin(fai), (2 * self._STEP, 1)).T.reshape(num)
        cos_fai = np.tile(np.cos(fai), (2 * self._STEP, 1)).T.reshape(num)
        x = sin_fai * cos_theta
        y = sin_fai * sin_theta
        z = cos_fai
        vector = np.array(list(zip(x, y, z)))

        # ridge_vertices为一维数组，保存每个Voronoi多边形的边界面对应的边角点
        # ridge_vertices_offset保存每个边界面ridge在ridge_vertices的起始索引
        # ralpha保存每个Voronoi多边形对应的边界极坐标函数，调用C++计算
        src_ridge_vertices_offset = np.cumsum([0, ] + [len(ridge) for ridge in src_voronoi.ridge_vertices])
        src_ridge_vertices = np.array(_flatten(src_voronoi.ridge_vertices))
        src_ralpha = np.zeros((self._src_particle_num, len(vector)))
        heuristic_dll.VoronoiPolar(src_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(vector),
                                   src_particle_ridges, src_particle_ridges_len, max_ridges,
                                   src_ridge_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   src_ridge_vertices_offset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   self._src_particle_num,
                                   src_particle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   src_voronoi.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        dest_ridge_vertices_offset = np.cumsum([0, ] + [len(ridge) for ridge in dest_voronoi.ridge_vertices])
        dest_ridge_vertices = np.array(_flatten(dest_voronoi.ridge_vertices))
        dest_ralpha = np.zeros((self._dest_particle_num, len(vector)))
        heuristic_dll.VoronoiPolar(dest_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(vector),
                                   dest_particle_ridges, dest_particle_ridges_len, max_ridges,
                                   dest_ridge_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   dest_ridge_vertices_offset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   self._dest_particle_num,
                                   dest_particle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   dest_voronoi.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        # 计算每个粒子对应的Voronoi相关系数，调用C++计算
        similarity_graph = np.zeros((self._src_particle_num, self._dest_particle_num), dtype=np.float64)
        heuristic_dll.VoronoiRelation(similarity_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      dest_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      src_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      len(vector), self._dest_particle_num, self._src_particle_num)

        '''
        np.save('similarity3d.npy', similarity_graph)
        similarity_graph = np.load('similarity3d.npy')
        '''
        # 过滤相关系数过小的值，目标函数为相关系数倒数
        similarity_graph = np.abs(similarity_graph)
        similarity_graph[similarity_graph < self._MIN_SIMILARITY] = self._MIN_SIMILARITY
        self._heuristic_graph = 1.0 / similarity_graph


class RelaxGraph(Heuristic3D.RelaxGraph):
    """用Voronoi作为邻居的判断条件，以Relaxation作为目标函数（JIA2015 RM-PTV），调用C++加速
    Reference: Relaxation algorithm-based PTV with dual calculation method and its application in addressing particle saltation
    """
    def _cal_graph(self):
        # 调用库函数直接求解Voronoi分割
        src_voronoi = scipy.spatial.Voronoi(self._src_particles)
        dest_voronoi = scipy.spatial.Voronoi(self._dest_particles)

        # 初始缓冲区大小为50，若不够按两倍扩容
        # 计算每个粒子按照Voronoi确定的邻居关系，调用C++计算
        max_neighbor = 50
        ret_flag = -1
        while ret_flag < 0:
            src_particle_neighbor = (ctypes.c_int * (max_neighbor * self._src_particle_num))()
            src_particle_neighbor_len = (ctypes.c_int * self._src_particle_num)()
            ret_flag = heuristic_dll.CalNeighbor(src_particle_neighbor, src_particle_neighbor_len, max_neighbor,
                                                 src_voronoi.ridge_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                 len(src_voronoi.ridge_points))
            dest_particle_neighbor = (ctypes.c_int * (max_neighbor * self._dest_particle_num))()
            dest_particle_neighbor_len = (ctypes.c_int * self._dest_particle_num)()
            ret_flag += heuristic_dll.CalNeighbor(dest_particle_neighbor, dest_particle_neighbor_len, max_neighbor,
                                                  dest_voronoi.ridge_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                  len(dest_voronoi.ridge_points))
            max_neighbor *= 2
        max_neighbor //= 2

        # 计算每个粒子对应的松弛长度，调用C++计算
        self._heuristic_graph = np.zeros((self._src_particle_num, self._dest_particle_num), dtype=np.float64)
        heuristic_dll.RelaxMethod(self._heuristic_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  dest_particle_neighbor, src_particle_neighbor,
                                  dest_particle_neighbor_len, src_particle_neighbor_len,
                                  max_neighbor, self._dest_particle_num, self._src_particle_num,
                                  self._dest_particles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  self._src_particles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))


class OldRelaxGraph(Heuristic3D.OldRelaxGraph):
    """用最紧邻法决定邻居对应关系，以Relaxation作为目标函数（Ohmi2010 ACO），调用C++加速
    Reference: Particle tracking velocimetry with an ant colony optimization algorithm
    """
    def _cal_graph(self):
        # 分别计算两帧中所有粒子的间距
        src = np.tile(self._src_particles, (self._src_particle_num, 1, 1))
        dest = np.tile(self._dest_particles, (self._dest_particle_num, 1, 1))
        src_dis = np.linalg.norm(src - np.swapaxes(src, 0, 1), axis=2)
        dest_dis = np.linalg.norm(dest - np.swapaxes(dest, 0, 1), axis=2)

        # self._MAX_NEIGHBOR确定选取的邻居数量，调用C++计算
        src_particle_neighbor = (ctypes.c_int * (self._MAX_NEIGHBOR * self._src_particle_num))()
        heuristic_dll.OldCalNeighbor(src_particle_neighbor,
                                     src_dis.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     self._MAX_NEIGHBOR, self._src_particle_num)
        dest_particle_neighbor = (ctypes.c_int * (self._MAX_NEIGHBOR * self._dest_particle_num))()
        heuristic_dll.OldCalNeighbor(dest_particle_neighbor,
                                     dest_dis.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     self._MAX_NEIGHBOR, self._dest_particle_num)

        # 计算每个粒子对应的松弛长度，调用C++计算
        self._heuristic_graph = np.zeros((self._src_particle_num, self._dest_particle_num), dtype=np.float64)
        heuristic_dll.OldRelaxMethod(self._heuristic_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     dest_particle_neighbor, src_particle_neighbor,
                                     self._MAX_NEIGHBOR, self._dest_particle_num, self._src_particle_num,
                                     self._dest_particles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     self._src_particles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))


class MixedGraph(Heuristic3D.MixedGraph):
    """将Distance与Voronoi目标函数混合，称为新混合目标函数，Voronoi部分调用C++加速

    成员变量：
        _distance：保存DistaceGraph类对象
        _voronoi：保存VoronoiGraph类对象
    """
    PARAMETER_NAMES = DistanceGraph.PARAMETER_NAMES.copy()
    PARAMETER_NAMES.update(VoronoiGraph.PARAMETER_NAMES)
    PARAMETER_NAMES['beta2'] = '混合比'

    def __init__(self, particles, parameter):
        self._distance = DistanceGraph(particles, parameter)
        self._voronoi = VoronoiGraph(particles, parameter)
        super(Heuristic3D.MixedGraph, self).__init__(particles, parameter)
