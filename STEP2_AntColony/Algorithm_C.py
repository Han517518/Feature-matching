"""
蚁群算法，求目标函数最优解，调用C++加速
"""

import enum
import ctypes
import numpy as np

import Algorithm


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    AntColony = '蚁群算法'


class AntC(ctypes.Structure):
    """c++中定义的蚂蚁类"""
    _fields_ = [('src_list', ctypes.POINTER(ctypes.c_int)),
                ('dest_list', ctypes.POINTER(ctypes.c_int)),
                ('total_distance', ctypes.c_double)]


# 指定用到的动态链接库，以及库内函数的参数表
algorithm_dll = ctypes.cdll.LoadLibrary(r'Algorithm.dll')
algorithm_dll.ACOCalStep.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(AntC),
                                     ctypes.POINTER(AntC), ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                     ctypes.c_int, ctypes.c_int)


class AntColony(Algorithm.AntColony):
    """蚁群算法实现，调用C++加速"""
    def calculate(self):
        # 计算概率矩阵
        self._update_possibility_graph()
        self._random_src()

        # 将蚂蚁转化为C++蚂蚁
        ant_cs = (AntC * self._ANT_NUM)()
        for i in range(self._ANT_NUM):
            self.ants[i].clear()
            ant_cs[i].src_list = self.ants[i].src_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            ant_cs[i].dest_list = self.ants[i].dest_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            ant_cs[i].total_distance = ctypes.c_double(self.ants[i].total_distance)
        best_ant_c = AntC(self.best_ant.src_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                          self.best_ant.dest_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                          ctypes.c_double(self.best_ant.total_distance))
        # 计算最优蚂蚁以及信息素增量，调用C++计算
        pheromone = np.zeros((self._src_graph_num, self._dest_graph_num), dtype=np.float64)
        algorithm_dll.ACOCalStep(pheromone.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 ctypes.pointer(best_ant_c),
                                 ant_cs, self._ANT_NUM,
                                 self._src_graph_num, self._dest_graph_num,
                                 self._possible_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 self._distance_graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 self._src_input_num, self._dest_input_num)
        for i in range(self._ANT_NUM):
            self.ants[i].total_distance = ant_cs[i].total_distance
        self.best_ant.total_distance = best_ant_c.total_distance

        # 更新信息素
        self._update_pheromone_graph(pheromone)

        return self.best_ant.src_index, self.best_ant.dest_index, self.best_ant.total_distance
