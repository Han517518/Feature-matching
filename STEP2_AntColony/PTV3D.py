"""
三维蚁群算法入口
"""

import importlib
import enum
import numpy as np
from numpy.lib.function_base import blackman
import scipy.stats

import ParticleGenerator3D


class Graph3DType(enum.Enum):
    Heuristic3D = '正常'
    Heuristic3D_C = 'C加速'


class SolverType(enum.Enum):
    Algorithm = '正常'
    Algorithm_Multi = '多进程'
    Algorithm_C = 'C加速'


class PTV3D(object):
    """计算三维PTV程序

    成员变量：
        _aco_module, _graph_module：蚁群算法和目标函数模块
        _aco_func, _graph_func：蚁群算法和目标函数的默认选择
        _default_betas：与_graph_module.Functions对应顺序的目标函数的默认beta值列表
        _parameter：预定义的所有algorithm和heuristic所用参数
    """
    def __init__(self):
        self._aco_module = importlib.import_module(SolverType.Algorithm_C.name)
        self._graph_module = importlib.import_module(Graph3DType.Heuristic3D_C.name)
        self._aco_func = getattr(self._aco_module, list(self._aco_module.Functions)[-1].name)
        self._graph_func = getattr(self._graph_module, list(self._graph_module.Functions)[-1].name)
        self._default_betas = [2.0, 6.0, 5.0, 5.0, 2.0]
        self._parameter = {
            'file_a': 'test_a.mat',
            'file_b': 'test_b.mat',
            'save_file': 'res.mat',
            'alpha': 1.0,
            'beta': 2.0,
            'beta2': 3.0,
            'rho': 0.5,
            'q': -1,
            'ant_num': 50,
            'random_src': True,
            'step': 15,
            'min_distance': 0.01,
            'min_similarity': 0.01,
            'max_neighbor': 5,
            'max_iter': 2000,
            'max_stable': 1500,
            'margin_rate': 0,
            'outer_iter': 10,
            'test': False,
            # 以下参数仅测试会用到
            'num': 1000,
            'straddle_frames': 4,
            'missing_rate': 0.2,
            'missing_bias': 0.25,
            'delta': 1,
            'record_generator': False,
            'lock_generator': False
        }

    def _load(self, generatorClass):
        """将三维数据读入并保存

        参数：
            generator_func：生成两帧三维坐标的函数，返回值格式为：src_particles, dest_particles，二维数组*_particles[*particle_num][3]
            *args：调用generator_func所需的参数
        """
        generator = generatorClass(self._parameter)
        self.src_particles, self.dest_particles = generator()
        self.src_particle_num = len(self.src_particles)
        self.dest_particle_num = len(self.dest_particles)
        self._num = max(self.src_particle_num, self.dest_particle_num)
        self._input_particles = min(self.src_particle_num, self.dest_particle_num)
        # 前后帧都出现粒子缺失，只有一部分粒子为可配对粒子（true_particle）
        if self._parameter['test']:
            self._true_particles = self._parameter['num'] - int(self._parameter['missing_rate']*self._parameter['num']) if isinstance(generator, ParticleGenerator3D.MissingFlow) else self._input_particles

    def _load_reverse(self):
        """将两帧粒子反序"""
        (self.src_particles, self.dest_particles, self.src_particle_num, self.dest_particle_num) = (self.dest_particles, self.src_particles, self.dest_particle_num, self.src_particle_num)

    def starter(self, cal_func, acoClass, graphClass):
        """不同计算函数启动器

        参数：
            cal_func：待启动的计算函数
            acoClass：选用蚁群算法类
            graphClass：选用目标函数类
        """
        margin_num = int(self._num * self._parameter['margin_rate'])
        n = cal_func.__code__.co_argcount
        if n == 3:
            # 判断参数个数，若为3则代表单向计算
            graph = graphClass((self.src_particles, self.dest_particles), self._parameter)
            graph.add_margin(margin_num)
            aco = acoClass((self.src_particle_num, self.dest_particle_num), self._parameter)
            aco.load_heuristic_graph(graph.get_heuristic_graph())
            forth, _ = cal_func(aco, graph)
            forth = list(zip(*forth))
            res = (np.array(forth[0]), np.array(forth[1]))
            aco.stop()
        elif n == 5:
            # 参数个数为5，代表应用了双向计算（或双向计算+反馈器）
            graph_forth = graphClass((self.src_particles, self.dest_particles), self._parameter)
            graph_forth.add_margin(margin_num)
            aco_forth = acoClass((self.src_particle_num, self.dest_particle_num), self._parameter)
            aco_forth.load_heuristic_graph(graph_forth.get_heuristic_graph())
            self._load_reverse()

            graph_back = graphClass((self.src_particles, self.dest_particles), self._parameter)
            graph_back.add_margin(margin_num)
            aco_back = acoClass((self.src_particle_num, self.dest_particle_num), self._parameter)
            aco_back.load_heuristic_graph(graph_back.get_heuristic_graph())
            self._load_reverse()

            forth, back = cal_func(aco_forth, graph_forth, aco_back, graph_back)
            output = list(map(lambda f, b: f == b, forth, back))
            res = np.array(forth)[output]
            res = (res[:, 0], res[:, 1])
            aco_forth.stop()
            aco_back.stop()
        else:
            res = (np.array([]), np.array([]))
        return res

    def _cal(self, aco, graph):
        """单向计算函数

        成员变量：
            _heuristic_best, _accuracy：一维数组，分别保存最优蚂蚁对应的目标函数值及正确率（用于测试）关于迭代次数变化

        返回值：
            src, dest：迭代完成后最优蚂蚁的两帧粒子对应关系

        参数：
            aco：已初始化的蚁群算法对象
            graph：已初始化的目标函数对象
        """
        self._heuristic_best = np.zeros(self._parameter['max_iter'])
        if self._parameter['test']:
            self._accuracy = np.zeros(self._parameter['max_iter'])
        virtual = 2 * np.mean(graph.get_heuristic_graph())
        graph.set_margin(virtual)

        count = 0
        old_distance = 0.0
        for i in range(self._parameter['max_iter']):
            src, dest, distance = aco.calculate()
            # print(distance)
            self._heuristic_best[i] = distance
            virtual = distance / self._num
            graph.set_margin(virtual)
            # 去除虚拟粒子
            true_particle = (src < self.src_particle_num) & (dest < self.dest_particle_num)
            src = src[true_particle]
            dest = dest[true_particle]
            if self._parameter['test']:
                # 计算本次迭代精度（src与dest对应位置粒子的id一致表明匹配正确，用于测试算法）
                self._accuracy[i] = sum((src == dest) & (src < self._true_particles)) / self._input_particles * 100
            # 若distance==old_distance，表示本轮迭代未优化，若未优化的轮数count达到max_stable，则判断收敛
            if abs(distance - old_distance) < old_distance * 1e-10:
                count += 1
                if count >= self._parameter['max_stable']:
                    break
            else:
                count = 0
            old_distance = distance
        # 若提早收敛，则将结果截断
        self._heuristic_best = self._heuristic_best[:i+1]
        if self._parameter['test']:
            self._accuracy = self._accuracy[:i+1]
        return list(zip(src, dest)), list(zip(dest, src))

    def _cal_dual(self, aco_forth, graph_forth, aco_back, graph_back):
        """双向计算函数

        成员变量：
            _net_accuracy：经过双向验证的净精度，即输出的正确粒子对除数量以输出的粒子对数量，其中输出指双向计算结果的交集部分
            _out_percentage：由于计算交集导致的输出率，定义为输出粒子对数量除以输入粒子对数量

        返回值：
            forth, back：正反两向的粒子匹配关系（每个变量为一个保存二元组列表，二元组为配对关系）

        参数：
            aco_forth, graph_froth：正向计算所用蚁群算法和目标函数对象
            aco_back, graph_back：反向计算所用蚁群算法和目标函数对象
        """
        # 先正反向应用两次_cal单向计算函数，并以前帧粒子id排序
        forth, _ = self._cal(aco_forth, graph_forth)
        forth.sort(key=lambda x: x[0])
        self._load_reverse()

        _, back = self._cal(aco_back, graph_back)
        back.sort(key=lambda x: x[0])
        self._load_reverse()

        # 统计双向配对交集的个数num，并进一步检查交集内正确配对的数量right
        i = 0
        j = 0
        num = 0
        right = 0
        while i < len(forth) and j < len(back):
            a = forth[i]
            b = back[j]
            if a[0] == b[0]:
                if a[1] == b[1]:
                    num += 1
                    if self._parameter['test']:
                        if a[0] == a[1] and a[0] < self._true_particles:
                            right += 1
                i += 1
                j += 1
            elif a[0] > b[0]:
                j += 1
            else:
                i += 1

        self._out_percentage = num / self._input_particles * 100
        if self._parameter['test']:
            self._net_accuracy = right / num * 100
        return forth, back

    def _cal_dual_phe(self, aco_forth, graph_forth, aco_back, graph_back):
        """信息素反馈的双向计算函数，返回值与参数和_cal_dual相同

        成员变量：
            _out_percentage：一维数组，输出率随外圈迭代的变化
            _net_accuracy：一维数组，净精度随外圈迭代的变化（用于测试）
        """
        total_num = self._num + int(self._num * self._parameter['margin_rate'])
        out_percentage = np.zeros(self._parameter['outer_iter'])
        if self._parameter['test']:
            net_accuracy = np.zeros(self._parameter['outer_iter'])

        # 每进行一次迭代，将反馈因子作用于信息素上
        for i in range(self._parameter['outer_iter']):
            forth, back = self._cal_dual(aco_forth, graph_forth, aco_back, graph_back)

            phe_forth = aco_forth.get_pheromone_graph()
            phe_back = aco_back.get_pheromone_graph()
            # 反馈因子的定义可参照reference
            self._feedback_factor = np.sqrt(
                2 * scipy.stats.mstats.gmean(scipy.stats.mstats.gmean(phe_forth)) * scipy.stats.mstats.gmean(scipy.stats.mstats.gmean(phe_back))
                / (np.sqrt(np.max(phe_forth) * np.min(phe_forth)) * np.sqrt(np.max(phe_back) * np.min(phe_back))))
            '''（已废弃）
            1 / 4 / np.sqrt(np.mean(1/phe_forth)*np.mean(1/phe_back) * 4
                                                / (np.min(1/phe_forth) + np.max(1/phe_forth))
                                                / (np.min(1/phe_back) + np.max(1/phe_back)))
            pow(np.min(phe_forth)*np.min(phe_back) / np.max(phe_forth) / np.max(phe_back), 0.25)
            '''
            feedback = np.ones([total_num, total_num]) * self._feedback_factor ** 2
            for path in forth:
                feedback[path[0]][path[1]] /= self._feedback_factor
            for path in back:
                feedback[path[0]][path[1]] /= self._feedback_factor

            aco_forth.set_feedback(feedback)
            aco_back.set_feedback(feedback.T)

            out_percentage[i] = self._out_percentage
            if self._parameter['test']:
                net_accuracy[i] = self._net_accuracy

        self._out_percentage = out_percentage
        if self._parameter['test']:
            self._net_accuracy = net_accuracy
        return forth, back

    def _cal_dual_heu(self, aco_forth, graph_forth, aco_back, graph_back):
        """启发因子反馈的双向计算函数，返回值与参数和_cal_dual相同

        成员变量：
            _out_percentage：一维数组，输出率随外圈迭代的变化
            _net_accuracy：一维数组，净精度随外圈迭代的变化（用于测试）
        """
        total_num = self._num + int(self._num * self._parameter['margin_rate'])
        out_percentage = np.zeros(self._parameter['outer_iter'])
        if self._parameter['test']:
            net_accuracy = np.zeros(self._parameter['outer_iter'])
        # 反馈因子的定义可参照reference
        self._feedback_factor = np.sqrt(2) \
            * np.sqrt(np.max(graph_forth.get_heuristic_graph()) * np.min(graph_forth.get_heuristic_graph())) \
            / scipy.stats.mstats.gmean(scipy.stats.mstats.gmean(graph_forth.get_heuristic_graph()))

        # 每进行一次迭代，将反馈因子作用于启发因子上
        for i in range(self._parameter['outer_iter']):
            aco_forth.load_heuristic_graph(graph_forth.get_heuristic_graph())
            aco_back.load_heuristic_graph(graph_back.get_heuristic_graph())
            forth, back = self._cal_dual(aco_forth, graph_forth, aco_back, graph_back)

            feedback = np.ones([total_num, total_num]) * self._feedback_factor ** 2
            for path in forth:
                feedback[path[0]][path[1]] /= self._feedback_factor
            for path in back:
                feedback[path[0]][path[1]] /= self._feedback_factor

            graph_forth.set_feedback(feedback)
            graph_back.set_feedback(feedback.T)

            out_percentage[i] = self._out_percentage
            if self._parameter['test']:
                net_accuracy[i] = self._net_accuracy

        # 在最后一次迭代后，沿用上轮信息素，清空启发因子反馈，再运行一次蚁群算法进行校正
        feedback = np.ones([total_num, total_num])
        graph_forth.set_feedback(feedback)
        graph_back.set_feedback(feedback.T)
        self._cal_dual(aco_forth, graph_forth, aco_back, graph_back)
        out_percentage[-1] = self._out_percentage
        if self._parameter['test']:
            net_accuracy[-1] = self._net_accuracy

        self._out_percentage = out_percentage
        if self._parameter['test']:
            self._net_accuracy = net_accuracy
        return forth, back

    def once(self, *input):
        """一个外部快速调用接口，input为输入粒子坐标"""
        self._load(lambda _: lambda: input)
        graph = self._graph_func((self.src_particles, self.dest_particles), self._parameter)
        graph.add_margin(int(self._num * self._parameter['margin_rate']))
        aco = self._aco_func((self.src_particle_num, self.dest_particle_num), self._parameter)
        aco.load_heuristic_graph(graph.get_heuristic_graph())
        forth, _ = self._cal(aco, graph)
        forth = list(zip(*forth))
        res = (np.array(forth[0]), np.array(forth[1]))
        aco.stop()
        return res

    def starter_all_funcs(self, cal_func, generatorClass, **kwargs):
        """启动全部计算函数用于横向对比"""
        def iterate_funcs(graph_func, beta):
            self._parameter['beta'] = beta
            self.starter(cal_func, self._aco_func, getattr(self._graph_module, graph_func.name))
            return self._accuracy[-1]
        self._parameter.update(kwargs)
        self._load(generatorClass)
        return list(map(iterate_funcs, self._graph_module.Functions, self._default_betas))

    def run(self):
        """启动程序并计算"""
        # """ 从MAT文件读入粒子坐标
        self._load(ParticleGenerator3D.LoadFlow)
        res = self.starter(self._cal_dual_heu, self._aco_func, self._graph_func)
        scipy.io.savemat(self._parameter['save_file'], {'src': res[0], 'dest': res[1]})
        # """
        """ 正常无粒子缺失流动
        for _ in range(30):
            self._load(ParticleGenerator3D.StdFlow)
            self.starter(self._cal, self._aco_func, self._graph_func)
            with open('res.txt', 'a') as fp:
                for acc in self._accuracy:
                    fp.write('%f\t' % acc)
                fp.write('\n')
        # """
        """ 无粒子缺失流动，用于测试混合比
        for beta1 in [0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            self._parameter['beta'] = beta1
            for beta2 in [0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
                for _ in range(30):
                    self._parameter['beta2'] = beta2/beta1
                    self._load(ParticleGenerator3D.StdFlow)
                    self.starter(self._cal, self._aco_func, self._graph_func)
                    with open('res.txt', 'a') as fp:
                        fp.write('%f\t' % (self._accuracy[-1]))
                        '''
                        for acc in self._accuracy:
                            fp.write('%f\t' % acc)
                        '''
                with open('res.txt', 'a') as fp:
                    fp.write('\n')
            with open('res.txt', 'a') as fp:
                fp.write('\n')
        # """
        """ 自制流场，用于测试比位移
        deltas = np.arange(0.1, 2.01, 0.1)
        res = [[self.starter_all_funcs(self._cal, ParticleGenerator3D.LabMadeFlow, delta=d) for _ in range(30)] for d in deltas]
        res = np.rollaxis(np.array(res), 2)
        with open('res.txt', 'a') as fp:
            for row in res:
                np.savetxt(fp, row, fmt='%f', delimiter='\t')
                fp.write('\n')
        # """
        """ 粒子缺失流场
        deltas = np.arange(0, 0.21, 0.025)
        for delta in deltas:
            self._parameter['missing_rate'] = delta
            for _ in range(5):
                self._load(ParticleGenerator3D.MissingFlow)
                self.starter(self._cal, self._aco_func, self._graph_func)
                with open('res.txt', 'a') as fp:
                    fp.write('%f\t' % (self._accuracy[-1]))
                    '''
                    for acc in self._accuracy:
                        fp.write('%f\t' % acc)s
                    '''
            with open('res.txt', 'a') as fp:
                fp.write('\n')
        with open('res.txt', 'a') as fp:
            fp.write('\n')
        # """
        """ 粒子缺失流场，目标函数横向测试
        deltas = np.arange(0, 0.21, 0.025)
        res = [[self.starter_all_funcs(self._cal, ParticleGenerator3D.MissingFlow, missing_rate=delta) for _ in range(30)] for delta in deltas]
        res = np.rollaxis(np.array(res), 2)
        with open('res.txt', 'a') as fp:
            for row in res:
                np.savetxt(fp, row, fmt='%f', delimiter='\t')
                fp.write('\n')
        # """
        """ 双向计算
        deltas = np.arange(0, 0.21, 0.025)
        for delta in deltas:
            self._parameter['missing_rate'] = delta
            for _ in range(30):
                self._load(ParticleGenerator3D.MissingFlow)
                self.starter(self._cal_dual, self._aco_func, self._graph_func)
                with open('res.txt', 'a') as fp:
                    fp.write('%f\t%f\n' % (self._net_accuracy, self._out_percentage))
            with open('res.txt', 'a') as fp:
                fp.write('\n')
        # """
        """ 信息素反馈
        for delta in np.arange(0, 0.21, 0.025):
            self._parameter['missing_rate'] = delta
            for _ in range(30):
                self._load(ParticleGenerator3D.MissingFlow)
                self.starter(self._cal_dual_phe, self._aco_func, self._graph_func)
                with open('res1.txt', 'a') as fp1, open('res2.txt', 'a') as fp2:
                    for net_acc, out_per in zip(self._net_accuracy, self._out_percentage):
                        fp1.write('%f\t' % net_acc)
                        fp2.write('%f\t' % out_per)
                    fp1.write('\n')
                    fp2.write('\n')
            with open('res1.txt', 'a') as fp1, open('res2.txt', 'a') as fp2:
                fp1.write('\n')
                fp2.write('\n')
        # """
        """ 启发因子反馈
        for delta in np.arange(0, 0.21, 0.025):
            for _ in range(30):
                self._parameter['missing_rate'] = delta
                self._load(ParticleGenerator3D.MissingFlow)
                self.starter(self._cal_dual_heu, self._aco_func, self._graph_func)
                with open('res3.txt', 'a') as fp1, open('res4.txt', 'a') as fp2:
                    for net_acc, out_per in zip(self._net_accuracy, self._out_percentage):
                        fp1.write('%f\t' % net_acc)
                        fp2.write('%f\t' % out_per)
                    fp1.write('\n')
                    fp2.write('\n')
            with open('res3.txt', 'a') as fp1, open('res4.txt', 'a') as fp2:
                fp1.write('\n')
                fp2.write('\n')
        # """
        """ 三维显示（已废弃）
        import matplotlib.pyplot as plt
        accfigure = plt.figure()
        accplot = accfigure.gca()
        accplot.plot(range(self._max_iter), self._accuracy)
        plt.xlabel('iteration', fontsize='x-large')
        plt.ylabel('Accuracy(%)', fontsize='x-large')
        plt.title('beta1=1    beta2=10', fontsize='xx-large')
        accplot.plot(beta2s, res)
        np.savetxt('res.txt', res, delimiter='\n')
        heufigure = plt.figure()
        heuplot = heufigure.gca()
        heuplot.plot(range(self._max_iter), self._heuristic_best)
        plt.xlabel('iteration', fontsize='x-large')
        plt.title('Fuction Value', fontsize='xx-large')
        plt.show()
        # """


if __name__ == '__main__':
    print("""
--------------------------------------------------------
    程序：蚁群PTV算法
    作者：聂铭远
    文件：""", __file__, """
    日期：2021-9-6
    语言：Python 3.7
--------------------------------------------------------
    """)
    ptv = PTV3D()
    ptv.run()
