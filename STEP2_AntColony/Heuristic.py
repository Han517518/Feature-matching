"""
将二维粒子坐标转化为特定的目标函数矩阵
"""

import enum
import numpy as np
import scipy.spatial
from functools import reduce
from itertools import repeat, chain, product


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    DistanceGraph = '最短位移'
    VoronoiGraph = '泰森最相似'
    RelaxGraph = '泰森松弛距离'
    OldRelaxGraph = '最紧邻松弛距离'
    MixedGraph = '混合判据'


class Graph(object):
    """所有计算目标函数的基类，提供了公共的调用接口

    类变量：
        PARAMETER_NAMES：保存了所需参数及对应描述（类常量），派生类应按需重写这个量

    成员变量：
        _heuristic_graph：保存计算完的目标函数值（OF）
        _feedback_graph：记录已应用的反馈矩阵（便于在下次反馈前撤销）
        _particle_src_num, _particle_dest_num: 不包含虚拟粒子的两帧粒子个数
        _graph_src_num, graph_dest_num：包含虚拟粒子的两帧粒子个数（始终表示矩阵的两维大小）
    """
    PARAMETER_NAMES = {}

    def __init__(self, particles, parameter):
        """初始化类，并调用计算函数启发因子_cal_graph"""
        super().__init__()
        self._src_particles, self._dest_particles = particles
        self._graph_src_num = self._src_particle_num = len(self._src_particles)
        self._graph_dest_num = self._dest_particle_num = len(self._dest_particles)
        self._feedback_graph = np.ones((self._graph_src_num, self._graph_dest_num))
        self._load_parameter(parameter)
        self._cal_graph()

    def _load_parameter(self, parameter):
        """将所需参数读入"""
        for name in self.PARAMETER_NAMES:
            setattr(self, '_'+name.upper(), parameter[name])

    def _cal_graph(self):
        """计算目标函数（OF）核心函数，后续不同派生类应重载本函数"""
        self._heuristic_graph = np.ones((self._src_particle_num, self._dest_particle_num))

    def add_margin(self, margin_num=50):
        """由于添加了虚拟粒子，将目标函数矩阵扩充
        若前帧粒子数大于后帧粒子数，则先将后帧加入虚拟粒子至：后帧粒子个数=前帧粒子个数
        此时输入满足条件：后帧粒子个数>=前帧粒子个数
        在此基础上，在后帧继续加入margin_num个虚拟粒子
        本算法暂时不支持自由调节前帧虚拟粒子数，前帧虚拟粒子数只能为添加前后帧粒子个数与前帧粒子个数之差

        参数：
            margin_num：指后帧粒子多加入的虚拟粒子数
        """
        # 补充后帧粒子
        max_num = max(self._src_particle_num, self._dest_particle_num)
        self._graph_dest_num = max_num + margin_num
        margin = np.ones((self._graph_src_num, self._graph_dest_num-self._dest_particle_num), dtype=np.float64)
        self._heuristic_graph = np.hstack((self._heuristic_graph, margin))
        self._feedback_graph = np.hstack((self._feedback_graph, margin))

    def set_margin(self, virtual):
        """将上述虚拟粒子设置虚拟目标函数vof"""
        self._heuristic_graph[self._src_particle_num:self._graph_src_num, :] = virtual
        self._heuristic_graph[:, self._dest_particle_num:self._graph_dest_num] = virtual

    def set_feedback(self, feedback):
        """将反馈矩阵feedback应用在_heuristic_graph上（应用前先撤销之前已应用的反馈矩阵_feedback_graph）"""
        self._heuristic_graph /= self._feedback_graph
        self._feedback_graph = feedback[:self._graph_src_num, :self._graph_dest_num]
        self._heuristic_graph *= self._feedback_graph

    def get_heuristic_graph(self):
        """返回计算结果_heuristic_graph"""
        return self._heuristic_graph


class DistanceGraph(Graph):
    """用最短距离作为目标函数"""
    PARAMETER_NAMES = Graph.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['min_distance'] = '最小距离'

    def _cal_graph(self):
        # 将各个坐标延展成矩阵，前帧为列向量按行展开，后帧为行向量按列展开（每个元素指一个n维坐标）
        src = np.tile(self._src_particles, (1, 1, self._dest_particle_num)).reshape((self._src_particle_num, self._dest_particle_num, -1))
        dest = np.tile(self._dest_particles, (self._src_particle_num, 1, 1))
        self._heuristic_graph = np.linalg.norm(dest - src, axis=2)
        # 过滤目标函数过小的值，防止计算启发因子（倒数）时过大
        self._heuristic_graph[self._heuristic_graph < self._MIN_DISTANCE] = self._MIN_DISTANCE


class VoronoiGraph(Graph):
    """用二维Voronoi多边形最相似作为目标函数（Zhang2015 VD-PTV）
    Reference: A particle tracking velocimetry algorithm based on the Voronoi diagram
    """
    PARAMETER_NAMES = Graph.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['step'] = '离散精度'
    PARAMETER_NAMES['min_similarity'] = '最小相关系数'

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

        def cal_polar_func(vertices):
            """根据给出的顶点坐标列表vertices，求解Voronoi对应的极坐标函数值（离散）"""
            # 计算矢量叉乘，若不满足顶点排序逆时针（角度递增），则将其反序
            if np.cross(vertices[0], vertices[1]) < 0.0:
                vertices = np.flipud(vertices)
            tran_vertices = vertices.T
            # 计算每个点对应的极径r和极角theta，判断y值正负，如果为负，theta = 360 - theta
            r = np.sqrt(np.square(tran_vertices[0]) + np.square(tran_vertices[1]))
            theta = np.arccos(tran_vertices[0] / r)
            theta_reverse = 2*np.pi - theta
            theta = np.where(tran_vertices[1] > 0.0, theta, theta_reverse)
            ralpha = np.zeros(self._STEP, dtype=np.float64)

            def ralpha_polar(p1, p2):
                """根据两极坐标点线段对应的极坐标函数（离散），p1p2为两点，p[0]为极径，p[1]为极角
                    直线方程为 r=r2r1sin(θ2-θ1)/[r1sin(θ-θ1)-r2sin(θ-θ2)]
                """
                a = p2[0] * p1[0] * np.sin(p2[1] - p1[1])
                section = (p2[1] > thetas) & (thetas > p1[1])
                ralpha[section] = a/(p1[0]*np.sin(thetas[section]-p1[1]) - p2[0]*np.sin(thetas[section]-p2[1]))

            # 将极坐标两个参数打包，并确保极角单调递增
            min_index = np.argmin(theta)
            polar_vertices = list(zip(r, theta))
            polar_vertices = polar_vertices[min_index:] + polar_vertices[:min_index]

            # 将顶点首尾相接，并至少覆盖[0,2pi)的范围
            list(map(ralpha_polar, chain([(polar_vertices[-1][0], polar_vertices[-1][1]-2*np.pi), ], polar_vertices), chain(polar_vertices, [(polar_vertices[0][0], polar_vertices[0][1]+2*np.pi), ])))
            return ralpha

        # vertices为三维数组迭代器，保存所有粒子（第一维）对应的Voronoi多边形的所有顶点相对于中心粒子坐标（第二维），每个坐标为一个二元素array（第三维）
        # 对两帧的Voronoi多边形计算边界面函数，由于前帧需要做平移，故后帧需要延展（可参照Zhang2015 VD-PTV论文）
        src_vertices = map(lambda region, p0: src_voronoi.vertices[src_voronoi.regions[region]] - p0, src_voronoi.point_region[:-len(margin_particle)], src_particle[:-len(margin_particle)])
        src_ralpha = list(map(cal_polar_func, src_vertices))
        dest_vertices = map(lambda region, p0: dest_voronoi.vertices[dest_voronoi.regions[region]] - p0, dest_voronoi.point_region[:-len(margin_particle)], dest_particle[:-len(margin_particle)])
        dest_ralpha = list(map(lambda vertices: np.tile(cal_polar_func(vertices), 2), dest_vertices))

        # 先计算src_ralpha, dest_ralpha的均值与标准差备用
        e_src_ralpha = map(lambda x: np.mean(x), src_ralpha)
        d_src_ralpha = map(lambda x: np.sqrt(np.var(x)), src_ralpha)
        e_dest_ralpha = [np.mean(x[:self._STEP]) for x in dest_ralpha]
        d_dest_ralpha = [np.sqrt(np.var(x[:self._STEP])) for x in dest_ralpha]

        # 计算相关系数，将src平移，与dest对应位置做互相关，取互相关最大值作为相关系数（可参照Zhang2015 VD-PTV论文）
        # 下列调用等价于（省略角标），k为平移距离，self._STEP为离散精度
        # for i for j
        #   similarity = (max(k=0,1,...,self._STEP-1)(Eij) - EiEj) / (σiσj)
        similarity_graph = np.array(list(map(
            lambda src, dest_r, e_src, e_dest_r, d_src, d_dest_r: list(map(
                lambda dest, e_dest, d_dest: (max([np.dot(src, dest[k:self._STEP+k]) for k in range(self._STEP)]) / self._STEP - e_src * e_dest) / (d_src * d_dest),
                dest_r, e_dest_r, d_dest_r)),
            src_ralpha, repeat(dest_ralpha), e_src_ralpha, repeat(e_dest_ralpha), d_src_ralpha, repeat(d_dest_ralpha))))
        '''
        np.save('similarity.npy', self.similarity_graph)
        similarity_graph = np.load('similarity.npy')
        '''
        # 过滤相关系数过小的值，目标函数为相关系数倒数
        similarity_graph[similarity_graph < self._MIN_SIMILARITY] = self._MIN_SIMILARITY
        self._heuristic_graph = 1.0 / similarity_graph


class RelaxGraph(Graph):
    """用Voronoi作为邻居的判断条件，以Relaxation作为目标函数（JIA2015 RM-PTV）
    Reference: Relaxation algorithm-based PTV with dual calculation method and its application in addressing particle saltation
    """
    def _cal_graph(self):
        # 调用库函数直接求解Voronoi分割
        src_voronoi = scipy.spatial.Voronoi(self._src_particles)
        dest_voronoi = scipy.spatial.Voronoi(self._dest_particles)

        def cal_neighbor(neighbor, ridge):
            """将ridge_points中保存的邻居对应关系拆解为某个粒子对应的全部邻居
                neighbor保存结果，二维数组，记载每个粒子对应的所有邻居
                ridge为Voronoi中认为是邻居的粒子对，[0][1]两个元素为邻居关系两端对应的粒子索引
            """
            neighbor[ridge[0]].append(ridge[1])
            neighbor[ridge[1]].append(ridge[0])

        src_particle_neighbor = [[] for _ in range(len(self._src_particles))]
        [cal_neighbor(src_particle_neighbor, ridge) for ridge in src_voronoi.ridge_points]
        dest_particle_neighbor = [[] for _ in range(len(self._dest_particles))]
        [cal_neighbor(dest_particle_neighbor, ridge) for ridge in dest_voronoi.ridge_points]

        def relax_func(src_neighbors, dest_neighbors, offset):
            """计算每个粒子对应的Relaxation（RF）
                src_neighbors, dest_neighbors两帧粒子的邻居对应的坐标，为二维坐标的数组
                offset两个粒子中心的位移
            """
            # 由少粒子像多粒子匹配
            if len(src_neighbors) > len(dest_neighbors):
                src_neighbors, dest_neighbors = (dest_neighbors, src_neighbors)
                offset = -offset

            # estimated_particles保存dest_neighbors按照offset逆位移后的位置，用于跟src_neighbors匹配
            estimated_particles = list(dest_neighbors - offset)

            def cal_min_distance(total_dis, p0):
                """寻找与位移后的dest_neighbors（即estimated_particles）距离最近的src_neighbors
                    贪心算法，每个src_neighbors都寻找其最近的estimated_particles，但每个estimated_particles只能被选择一次
                    total_dis为总距离，迭代法求总距离=之前的总距离+本次距离
                    p0本次迭代src_neighbors的坐标
                """
                dis = np.linalg.norm(estimated_particles - p0, ord=2, axis=1)
                min_index = dis.argmin()
                estimated_particles.pop(min_index)
                return total_dis + dis[min_index]

            return reduce(cal_min_distance, src_neighbors, 0.0)

        # 先行计算每个粒子的中心位移offset
        # 下列调用等价于（省略角标）
        # for i for j
        #   offset=self._dest_particles-self._src_particles
        #   self._heuristic_graph=relax_fuc(self._src_particles[src_particle_neighbor], self._dest_particles[dest_particle_neighbor], offset)
        offsets = map(np.subtract, repeat(self._dest_particles), self._src_particles)
        self._heuristic_graph = np.array(list(map(
            lambda src_indexes, dest_particles, offset: list(map(
                relax_func, repeat(self._src_particles[src_indexes]), dest_particles, offset)),
            src_particle_neighbor, repeat([self._dest_particles[indexes] for indexes in dest_particle_neighbor]), offsets)))


class OldRelaxGraph(Graph):
    """用最紧邻法决定邻居对应关系，以Relaxation作为目标函数（Ohmi2010 ACO）
    Reference: Particle tracking velocimetry with an ant colony optimization algorithm
    """
    PARAMETER_NAMES = Graph.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['max_neighbor'] = '邻居个数'

    def _cal_graph(self):
        # 分别计算两帧中所有粒子的间距
        src = np.tile(self._src_particles, (self._src_particle_num, 1, 1))
        dest = np.tile(self._dest_particles, (self._dest_particle_num, 1, 1))
        src_dis = np.linalg.norm(src - np.swapaxes(src, 0, 1), axis=2)
        dest_dis = np.linalg.norm(dest - np.swapaxes(dest, 0, 1), axis=2)

        # 选择的最近邻居
        src_particle_neighbor = np.argpartition(src_dis, self._MAX_NEIGHBOR + 1)
        dest_particle_neighbor = np.argpartition(dest_dis, self._MAX_NEIGHBOR + 1)

        # 将n个最近的邻居按从小到大重新排序，并剔除最近的粒子（粒子到其本身的距离为0）
        src_particle_neighbor_sorted = map(lambda neighbor, dis: sorted(neighbor[:self._MAX_NEIGHBOR+1], key=lambda index: dis[index])[1:], src_particle_neighbor, src_dis)
        dest_particle_neighbor_sorted = list(map(lambda neighbor, dis: sorted(neighbor[:self._MAX_NEIGHBOR+1], key=lambda index: dis[index])[1:], dest_particle_neighbor, dest_dis))

        # 先行计算每个粒子的中心位移offset
        # 下列调用等价于（省略角标）
        # for i for j
        #   offset=self._dest_particles-self._src_particles
        #   self._heuristic_graph=sum(k=0,1,...,self._MAX_NEIGHBOR-1)(self._src_particles[src_particle_neighbor] - (self._dest_particles[dest_particle_neighbor] - offset))
        offsets = map(np.subtract, repeat(self._dest_particles), self._src_particles)
        self._heuristic_graph = np.array(list(map(
            lambda src_indexes, dest_particles, offset: list(map(
                lambda src, dest, off: sum(np.linalg.norm(dest - off - src, ord=2, axis=1)),
                repeat(self._src_particles[src_indexes]), dest_particles, offset)),
            src_particle_neighbor_sorted, repeat([self._dest_particles[indexes] for indexes in dest_particle_neighbor_sorted]), offsets)))


class MixedGraph(Graph):
    """将Distance与Voronoi目标函数混合，称为新混合目标函数

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
        super().__init__(particles, parameter)

    def _cal_graph(self):
        self._heuristic_graph = self._distance.get_heuristic_graph() * self._voronoi.get_heuristic_graph() ** self._BETA2
