"""
将二维粒子坐标转化为特定的目标函数矩阵，调用C++加速
"""

import enum
import ctypes
import numpy as np
import scipy.spatial
from itertools import product
from tkinter import _flatten

import Heuristic


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    DistanceGraph = '最短位移'
    VoronoiGraph = '泰森最相似'
    RelaxGraph = '泰森松弛距离'
    OldRelaxGraph = '最紧邻松弛距离'
    MixedGraph = '混合判据'


# 指定用到的动态链接库，以及库内函数的参数表
heuristic_dll = ctypes.cdll.LoadLibrary(r'Heuristic.dll')
heuristic_dll.VoronoiPolar.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64),
                                       ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
heuristic_dll.VoronoiRelation.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
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


class DistanceGraph(Heuristic.DistanceGraph):
    """用三维最短距离作为目标函数，提供统一接口，并未加速"""
    pass


class VoronoiGraph(Heuristic.VoronoiGraph):
    """用二维Voronoi多边形最相似作为目标函数（Zhang2015 VD-PTV），调用C++加速
    Reference: A particle tracking velocimetry algorithm based on the Voronoi diagram
    """
    def _cal_graph(self):
        # 四个边角点加入，用于防止边角处Voronoi多边形不封闭
        pos_min = np.minimum(np.min(self._src_particles, axis=0), np.min(self._dest_particles, axis=0)) - 1e-5
        pos_max = np.maximum(np.max(self._src_particles, axis=0), np.max(self._dest_particles, axis=0)) + 1e-5
        margin_particle = np.array(list(product(*zip(pos_min, pos_max))))
        src_particle = np.vstack((self._src_particles, margin_particle))
        dest_particle = np.vstack((self._dest_particles, margin_particle))

        # 调用库函数直接求解Voronoi分割
        src_voronoi = scipy.spatial.Voronoi(src_particle)
        dest_voronoi = scipy.spatial.Voronoi(dest_particle)
        # 确定离散自变量
        thetas = np.arange(0, 2*np.pi, 2*np.pi/self._STEP)

        # region_vertices为一维数组，保存每个Voronoi多边形对应的边角点
        # regions_lensum保存每个region在region_vertices的终止索引（该多边形及多边形之前的所有累计长度和）
        # ralpha保存每个Voronoi多边形对应的边界极坐标函数，调用C++计算
        src_regions_lensum = np.cumsum([len(region) for region in src_voronoi.regions])
        src_region_vertices = np.array(_flatten(src_voronoi.regions))
        src_ralpha = np.zeros((self._src_particle_num, self._STEP), dtype=np.float64)
        heuristic_dll.VoronoiPolar(src_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   thetas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._STEP,
                                   src_region_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   src_regions_lensum.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   src_voronoi.point_region.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                   self._src_particle_num,
                                   src_particle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   src_voronoi.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        dest_regions_lensum = np.cumsum([len(region) for region in dest_voronoi.regions])
        dest_region_vertices = np.array(_flatten(dest_voronoi.regions))
        dest_ralpha = np.zeros((self._dest_particle_num, self._STEP), dtype=np.float64)
        heuristic_dll.VoronoiPolar(dest_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   thetas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._STEP,
                                   dest_region_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   dest_regions_lensum.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   dest_voronoi.point_region.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                   self._dest_particle_num,
                                   dest_particle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   dest_voronoi.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        # 计算每个粒子对应的Voronoi相关系数，调用C++计算
        similarity_graph = np.zeros((self._src_particle_num, self._dest_particle_num), dtype=np.float64)
        heuristic_dll.VoronoiRelation(similarity_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      dest_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      src_ralpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                      self._STEP, self._dest_particle_num, self._src_particle_num)
        # 过滤相关系数过小的值，目标函数为相关系数倒数
        similarity_graph[similarity_graph < self._MIN_SIMILARITY] = self._MIN_SIMILARITY
        self._heuristic_graph = 1.0 / similarity_graph


class RelaxGraph(Heuristic.RelaxGraph):
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


class OldRelaxGraph(Heuristic.OldRelaxGraph):
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


class MixedGraph(Heuristic.MixedGraph):
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
        super(Heuristic.MixedGraph, self).__init__(particles, parameter)
