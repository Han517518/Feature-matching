"""
将三维粒子坐标转化为特定的目标函数矩阵
"""

import enum
import numpy as np
import scipy.spatial
from itertools import product

import Heuristic


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    DistanceGraph = '最短位移'
    VoronoiGraph = '泰森最相似'
    RelaxGraph = '泰森松弛距离'
    OldRelaxGraph = '最紧邻松弛距离'
    MixedGraph = '混合判据'


class DistanceGraph(Heuristic.DistanceGraph):
    """用最短距离作为目标函数，与二维计算方法相同"""
    pass


class VoronoiGraph(Heuristic.Graph):
    """用三维Voronoi多面体最相似作为目标函数（Nie2021 DPF-PTV）
    Reference: A hybrid 3D particle matching algorithm based on ant colony optimization
    """
    PARAMETER_NAMES = Heuristic.Graph.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['step'] = '离散精度'
    PARAMETER_NAMES['min_similarity'] = '最小相关系数'

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

        def cal_ridge(particle_ridges, ridge, ridge_index):
            """根据边界面对应的两侧粒子索引ridge，将边界面添加至各粒子的Voronoi多面体边界面索引数组particle_ridge中"""
            particle_ridges[ridge[0]].append(ridge_index)
            particle_ridges[ridge[1]].append(ridge_index)

        src_particle_ridges = [[] for _ in range(len(src_particle))]
        [cal_ridge(src_particle_ridges, ridge, index) for index, ridge in enumerate(src_voronoi.ridge_points)]
        dest_particle_ridges = [[] for _ in range(len(dest_particle))]
        [cal_ridge(dest_particle_ridges, ridge, index) for index, ridge in enumerate(dest_voronoi.ridge_points)]

        def pre_cal_ray(step):
            """先行计算单位方向矢量（自变量的离散值），以中心粒子为原点，发射的该向量组与其对应的Voronoi多面体的交点为待求点（可参照Nie2021 DPF-PTV论文）
            求得的方向矢量保存在待修饰函数的vector成员内"""
            def decorate(func):
                num = 2*step*step
                theta = np.arange(0, 2 * np.pi, np.pi / step)
                fai = np.arange(0.5 * np.pi / step, np.pi, np.pi / step)
                sin_theta = np.tile(np.sin(theta), step)
                cos_theta = np.tile(np.cos(theta), step)
                sin_fai = np.tile(np.sin(fai), (2 * step, 1)).T.reshape(num)
                cos_fai = np.tile(np.cos(fai), (2 * step, 1)).T.reshape(num)
                x = sin_fai * cos_theta
                y = sin_fai * sin_theta
                z = cos_fai
                vector = np.array(list(zip(x, y, z)))
                setattr(func, 'vector', vector)
                return func
            return decorate

        def cal_polar_func(vertices):
            """根据一个粒子对应的Voronoi多面体的各个边界面，计算球坐标函数（离散）
                vertices描述了一个Voronoi多面体，第一维表示各个边界面，第二维代表各个边界面中包含的所有边角顶点
                自变量（方向向量）由于被修饰，保存在cal_polar_func.vector中
            """
            def point_3d_2d(points, ignore_dim):
                """将一个三维点组points按照ignore_dim所指示的方向投影成二维点，并在该二维平面上按照逆时针排序"""
                p = np.delete(points, ignore_dim, axis=1)
                p0 = np.mean(p, axis=0)

                def sort_fun(point):
                    """计算边缘点point到中心p0的角度的cos值（角度小于180的部分沿y=-1翻转，使得该函数单调递增，值域(-3,1)）"""
                    vec = point - p0
                    r = np.linalg.norm(vec, ord=2)
                    return -2 - vec[0]/r if vec[1] > 0 else vec[0]/r

                return sorted(p, key=sort_fun)

            # 计算平面的若干参数，平面法线n_plane，平面参数d_plane，投影方向ignore_dim，投影点vertices_2d
            # 平面方程为(n1)x + (n2)y + (n3)z + d = 0
            n_plane = list(map(lambda vertices: np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1]), vertices))
            d_plane = list(map(lambda n, vertices: -np.dot(n, vertices[0]), n_plane, vertices))
            ignore_dim = np.abs(n_plane).argmax(axis=1)
            vertices_2d = list(map(point_3d_2d, vertices, ignore_dim))

            def cross2d(vec1, vec2):
                """两个二维矢量叉乘"""
                return vec1[0] * vec2[1] - vec1[1] * vec2[0]

            def is_inside(p, vert):
                """二维点p是否在由顶点组vert构成的二维物理平面（边界条件）内"""
                n = len(vert)
                # 以vert[0]作为原点，叉乘大于0代表point在直线v0v1右侧，不满足（因为多边形为逆时针排列顶点）
                if cross2d(p - vert[0], vert[1] - vert[0]) > 0:
                    return False
                # 同理，如果point在直线v0v(n-1)左侧，也不满足
                if cross2d(p - vert[0], vert[n-1] - vert[0]) < 0:
                    return False
                # 如果在这两之间，则需要找到一个最小的扇形，三边分别为v0v(line-1),v0v(line)和无穷远，使得point在这个扇形内
                start = 2
                end = n-1
                line = -1
                while start <= end:
                    mid = (start + end) // 2
                    if cross2d(p - vert[0], vert[mid] - vert[0]) > 0:
                        line = mid
                        end = mid - 1
                    else:
                        start = mid + 1
                # 最后只需验证point是否在直线v(line-1)v(line）左边即可
                return cross2d(p - vert[line-1], vert[line] - vert[line-1]) <= 0

            def vector_polar(vector):
                """在确定边界面参数后，计算球坐标函数（离散）"""
                # 遍历每个边界面，即对应的平面四个元素
                for vert_2d, n, d, ign in zip(vertices_2d, n_plane, d_plane, ignore_dim):
                    # 根据方向矢量计算交点以及对应极径
                    try:
                        rho = -d / np.dot(n, vector)
                    except ZeroDivisionError:
                        continue
                    except RuntimeWarning:
                        continue
                    # 若交点在方向的反向，则放弃
                    if rho <= 0.0:
                        continue
                    # 若交点在边界面内，则终止迭代（与其他边界面必不相交），返回极径值
                    if is_inside(np.delete(rho*vector, ign), vert_2d):
                        return rho
                        ''' 三维显示（已废除）
                        import matplotlib as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        figure2d = plt.figure()
                        figure3d = Axes3D(figure2d)
                        xx = []
                        yy = []
                        zz = []
                        for point in plane:
                            xx.append(point[0])
                            yy.append(point[1])
                            zz.append(point[2])
                        xx.append(plane[0][0])
                        yy.append(plane[0][1])
                        zz.append(plane[0][2])
                        figure3d.plot(xx, yy, zz, c='b')
                        figure3d.scatter(rho*v[0], rho*v[1], rho*v[2], c='b')
                        xx = []
                        yy = []
                        zz = []
                        for point in old_plane:
                            xx.append(point[0])
                            yy.append(point[1])
                            zz.append(point[2])
                        xx.append(old_plane[0][0])
                        yy.append(old_plane[0][1])
                        zz.append(old_plane[0][2])
                        figure3d.plot(xx, yy, zz, c='r')
                        figure3d.scatter(old_point[0], old_point[1], old_point[2], c='r')
                        plt.show()
                    old_plane = plane
                    old_point = rho*v
                    old_ig = ignore_dim
                    '''
                # 正常情况下不应返回0.0，仅为了保证返回浮点数
                return 0.0

            # 计算该Voronoi多面体对应与所有方向向量交点的极径
            return [vector_polar(vector) for vector in cal_polar_func.vector]

        # 确定离散自变量
        cal_polar_func = pre_cal_ray(self._STEP)(cal_polar_func)

        # vertices为四维数组迭代器，保存所有粒子（第一维）对应的Voronoi多面体的所有边界面（第二维）的所有顶点相对于中心粒子坐标（第三维），每个坐标为一个三元素array（第四维）
        # 对两帧的Voronoi多边形计算边界面函数
        src_vertices = map(lambda ridges, p0: list(map(lambda ridge: src_voronoi.vertices[src_voronoi.ridge_vertices[ridge]] - p0, ridges)), src_particle_ridges[:-len(margin_particle)], src_particle[:-len(margin_particle)])
        src_ralpha = np.array(list(map(cal_polar_func, src_vertices)))
        dest_vertices = map(lambda ridges, p0: list(map(lambda ridge: dest_voronoi.vertices[dest_voronoi.ridge_vertices[ridge]] - p0, ridges)), dest_particle_ridges[:-len(margin_particle)], dest_particle[:-len(margin_particle)])
        dest_ralpha = np.array(list(map(cal_polar_func, dest_vertices)))

        # 先计算ralpha的均值与标准差备用
        e_src_ralpha = map(lambda x: np.mean(x), src_ralpha)
        d_src_ralpha = map(lambda x: np.sqrt(np.var(x)), src_ralpha)
        e_dest_ralpha = np.array([np.mean(x) for x in dest_ralpha])
        d_dest_ralpha = np.array([np.sqrt(np.var(x)) for x in dest_ralpha])

        # 计算src与dest的相关系数，下列调用等价于（省略角标）
        # for i for j
        #   similarity = (Eij - EiEj) / (σiσj)
        similarity_graph = np.array(list(map(
            lambda src, e_src, d_src: (np.dot(dest_ralpha, src) / len(src) - e_src * e_dest_ralpha) / (d_src * d_dest_ralpha),
            src_ralpha, e_src_ralpha, d_src_ralpha)))
        '''
        np.save('similarity3d.npy', similarity_graph)
        similarity_graph = np.load('similarity3d.npy')
        '''
        # 过滤相关系数过小的值，目标函数为相关系数倒数
        similarity_graph = np.abs(similarity_graph)
        similarity_graph[similarity_graph < self._MIN_SIMILARITY] = self._MIN_SIMILARITY
        self._heuristic_graph = 1.0 / similarity_graph


class RelaxGraph(Heuristic.RelaxGraph):
    """用Voronoi作为邻居的判断条件，以Relaxation作为目标函数（JIA2015 RM-PTV）
    Reference: Relaxation algorithm-based PTV with dual calculation method and its application in addressing particle saltation
    与二维计算方法相同
    """
    pass


class OldRelaxGraph(Heuristic.OldRelaxGraph):
    """用最紧邻法决定邻居对应关系，以Relaxation作为目标函数（Ohmi2010 ACO）
    Reference: Particle tracking velocimetry with an ant colony optimization algorithm
    与二维计算方法相同
    """
    pass


class MixedGraph(Heuristic.Graph):
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
