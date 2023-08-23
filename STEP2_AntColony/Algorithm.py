"""
蚁群算法，求目标函数最优解
"""

import enum
import copy
import random
import numpy as np
from functools import reduce


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    AntColony = '蚁群算法'


class ACOBase(object):
    """蚁群算法的基类，提供了公共的调用接口

    类变量：
        PARAMETER_NAMES：保存了所需参数及对应描述（类常量），派生类应按需重写这个量

    成员变量：
        _src_input_num, dest_input_num：两帧输入粒子的数量（不含虚拟粒子）
        _src_graph_num, _dest_graph_num：两帧粒子的数量（含虚拟粒子，与_distance_graph大小一致）
        _distance_graph：粒子对应目标函数矩阵
        _heuristic_graph：启发因子矩阵，为目标函数倒数
        _pheromone_graph：信息素矩阵，反应蚂蚁的群体经验
        _possible_graph：每个路径的访问概率矩阵，为_heuristic_graph和_pheromone_graph的加权积
    """
    PARAMETER_NAMES = {}

    def __init__(self, particle_size, parameter):
        """初始化类，并调用两个重载函数_load_parameter和_create_ants

        参数：
            parameter：用于派生类的自定义参数
            particle_size：表明两帧粒子个数的二元组
        """
        self._src_input_num, self._dest_input_num = particle_size
        self._src_graph_num, self._dest_graph_num = particle_size
        self._heuristic_graph = np.ones(particle_size, dtype=np.float64)
        self._distance_graph = np.ones(particle_size, dtype=np.float64)
        self._pheromone_graph = np.ones(particle_size, dtype=np.float64)
        self._possible_graph = np.ones(particle_size, dtype=np.float64)
        self._load_parameter(parameter)
        self._create_ants()
        super().__init__()

    def _load_parameter(self, parameter):
        """将所需参数读入"""
        for name in self.PARAMETER_NAMES:
            setattr(self, '_'+name.upper(), parameter[name])
        pass

    def _create_ants(self):
        """派生类应重载此函数创建不同类的蚂蚁"""
        pass

    def load_heuristic_graph(self, graph):
        """读入启发因子矩阵"""
        self._distance_graph = graph
        self._heuristic_graph = 1.0 / graph
        self._src_graph_num, self._dest_graph_num = np.shape(graph)

    def calculate(self):
        """核心计算函数，派生类应重载此函数实现不同功能"""
        return [], [], 0.0

    def stop(self):
        """用于收尾的函数，派生类可重载此函数用于收尾工作"""
        pass

    def get_pheromone_graph(self):
        """返回信息素矩阵"""
        return self._pheromone_graph

    def set_feedback(self, feedback):
        """将反馈矩阵应用在信息素矩阵上"""
        self._pheromone_graph *= feedback[:self._src_graph_num][:self._dest_graph_num]


class Ant(object):
    """描述一只蚂蚁个体的类

    成员变量：
        _id：该蚂蚁的顺序号
        _src_num, _dest_num：前后两帧粒子个数
        src_index, dest_index：前后两帧粒子顺序索引，相同位置的粒子认为是配对粒子
        total_distace：蚂蚁走过总路程
        _move_count：蚂蚁已移动过的步数
        _available_flag：蚂蚁可以访问的粒子标记（已访问过的粒子不能重复访问）
    """
    def __init__(self, idn, src_num, dest_num):
        """初始化类，并调用clear初始化另一部分成员"""
        self._id = idn
        self._src_num = src_num
        self._dest_num = dest_num
        self.src_index = np.arange(src_num)
        self.clear()
        super().__init__()

    def clear(self):
        """在每轮迭代后，重新初始化部分成员"""
        self.dest_index = np.zeros(self._src_num, dtype=np.int32)
        self.total_distance = 0.0
        self._move_count = 0
        self._available_flag = np.ones(self._dest_num)

    def _next_point(self, possible_graph):
        """根据访问概率和访问标记，按轮盘法则选择一个粒子"""
        # src_index[_move_count]表示待配对的前帧粒子，计算其与后帧所有粒子的配对可能性
        # prob_graph为配对可能性的累加和，即与本粒子与之前所有粒子的配对可能性总和
        # 轮盘赌法确定待配对粒子，产生一个随机概率，0.0-prob_sum，判断这个概率所在的区间
        point = -1
        prob_graph = np.cumsum(self._available_flag * possible_graph[self.src_index[self._move_count]])
        prob_sum = prob_graph[-1]
        if prob_sum > 0.0:
            prob = random.uniform(0.0, prob_sum)
            point = np.argmax(prob_graph > prob)

        if point == -1:     # 如果按上述轮盘赌没找到粒子（正常情况下，程序不应满足此条件）
            point = random.randint(0, self._dest_num - 1)
            if sum(self._available_flag) > 0:   # 还有剩余粒子可供配对
                while (self._available_flag[point]) == 0.0:  # if==False，说明已经访问过了，从未访问的粒子中随机选一个
                    point = random.randint(0, self._dest_num - 1)
            else:   # 没有剩余粒子，随机输出一个
                print('Warning! There is no particle candidate remaining.')

        return point

    def cal_total_distance(self, distance_graph, particle_size):
        """根据已经配对的关系，计算蚂蚁走过的总路程，particle_size为实际输入的粒子数量二元组（不含虚拟粒子）"""
        self.total_distance = reduce(lambda dis, path: (dis + distance_graph[path[0]][path[1]]) if path[1] < particle_size[1] and path[0] < particle_size[0] else dis, zip(self.src_index, self.dest_index), 0.0)

    def _move(self, point):
        """将选择好的粒子放入序列，并更新访问列表"""
        self.dest_index[self._move_count] = point
        self._available_flag[point] = 0.0
        self._move_count += 1

    def search_path(self, possible_graph):
        """蚂蚁的对外接口，负责开启一轮计算"""
        # 初始化数据
        self.clear()

        # 搜素路径，遍历完所有粒子为止
        while self._move_count < self._src_num:
            # 根据概率选择粒子，并移动移动
            point = self._next_point(possible_graph)
            self._move(point)

    def random_src(self):
        """提供蚂蚁随机跳回的处理，将_src_index乱序"""
        self.src_index = np.random.permutation(self._src_num)


class AntColony(ACOBase):
    """蚁群算法实现"""
    PARAMETER_NAMES = ACOBase.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['alpha'] = '信息素权重'
    PARAMETER_NAMES['beta'] = '启发因子权重'
    PARAMETER_NAMES['rho'] = '信息素挥发速率'
    PARAMETER_NAMES['q'] = '信息素增量Q常数'
    PARAMETER_NAMES['ant_num'] = '蚂蚁数量'
    PARAMETER_NAMES['random_src'] = '是否乱序(True/False)'

    def load_heuristic_graph(self, graph):
        """读入启发因子矩阵
            graph在此算法为目标函数矩阵，与_distance_graph对应
            _heuristic_graph启发因子矩阵，为目标函数的倒数（预先乘混合比_BETA）
            更新两帧粒子的数量_src_graph_num, _dest_graph_num与_distance_graph大小一致）
            初始化信息素矩阵_pheromone_graph和概率矩阵_possible_graph
        """
        self._distance_graph = graph
        self._heuristic_graph = 1.0 / graph ** self._BETA
        self._src_graph_num, self._dest_graph_num = np.shape(graph)
        self._pheromone_graph = np.ones(np.shape(graph), dtype=np.float64)
        self._possible_graph = np.ones(np.shape(graph), dtype=np.float64)
        self._create_ants()

    def _create_ants(self):
        """创建规定数量的蚂蚁，并定义一个最优蚂蚁（存储一轮迭代中目标函数和最小的蚂蚁）"""
        self.ants = [Ant(i, self._src_graph_num, self._dest_graph_num) for i in range(self._ANT_NUM)]
        self.best_ant = Ant(-1, self._src_graph_num, self._dest_graph_num)
        self.best_ant.dest_index = np.random.permutation(self._src_graph_num)
        self.best_ant.total_distance = np.inf

    def calculate(self):
        """蚁群算法核心计算函数"""
        # 计算概率矩阵，并更新最优蚂蚁的目标函数和（由于反馈机制，不同迭代的_distance_graph可能不一致）
        self._update_possibility_graph()
        self.best_ant.cal_total_distance(self._distance_graph, (self._src_input_num, self._dest_input_num))
        self._random_src()

        for ant in self.ants:
            ant.search_path(self._possible_graph)
            # 计算路径总长度
            ant.cal_total_distance(self._distance_graph, (self._src_input_num, self._dest_input_num))
            # 更新最优解
            if ant.total_distance < self.best_ant.total_distance:
                self.best_ant = copy.deepcopy(ant)

        print(self.best_ant.total_distance)
        # 更新信息素
        pheromone = np.zeros((self._src_graph_num, self._dest_graph_num), dtype=np.float64)
        for ant in self.ants:
            # 信息素增量dp与路径总距离反比
            dp = 1.0 / (ant.total_distance)

            def add_pheromone(src, dest):
                pheromone[src][dest] += dp

            list(map(add_pheromone, ant.src_index, ant.dest_index))
        self._update_pheromone_graph(pheromone)

        return self.best_ant.src_index, self.best_ant.dest_index, self.best_ant.total_distance

    def _update_pheromone_graph(self, pheromone):
        '''更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素'''
        self._pheromone_graph = self._RHO * self._pheromone_graph + self._get_q(pheromone) * pheromone * (1-self._RHO)

    def _update_possibility_graph(self):
        """更新概率矩阵"""
        self._heuristic_graph = 1.0 / self._distance_graph ** self._BETA
        self._possible_graph = self._pheromone_graph ** self._ALPHA * self._heuristic_graph

    def _random_src(self):
        """对所有蚂蚁应用随机跳回"""
        if self._RANDOM_SRC:
            for ant in self.ants:
                ant.random_src()

    def _get_q(self, pheromone):
        """返回信息素增量常数Q"""
        if self._Q == -1.0:
            # 若Q==-1.0表明采用自适应Q
            return 1.0 / np.max(pheromone)
        return self._Q
