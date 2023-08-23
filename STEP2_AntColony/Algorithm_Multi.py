"""
蚁群算法，求目标函数最优解，Python多进程加速
"""

import enum
import copy
import multiprocessing
import numpy as np

import Algorithm


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    AntColony = '蚁群算法'


class AntColony(Algorithm.AntColony):
    """蚁群算法实现，Python多进程加速

    成员变量：
        _max_core：本算法创立进程数，CPU支持的最大同时运行线程数
        _task_queue：向子进程分配任务的队列
        _result_queue：计算完成的返回队列
        _queue_lock：用于子进程间访问队列的锁
        _process：所有子进程列表
        _command_pipes：分别控制所有子进程的指令管道列表
    """
    class CommandType(enum.IntEnum):
        """定义控制子进程的指令"""
        Exit = 0
        LoadDistance = 1
        LoadVirtual = 2
        LoadPossible = 3
        Calculate = 4

    def __init__(self, particle_size, parameter):
        super().__init__(particle_size, parameter)
        self._max_core = multiprocessing.cpu_count()
        self._task_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._queue_lock = multiprocessing.RLock()
        self._process = []
        self._command_pipes = []

    def __del__(self):
        self.stop()

    @staticmethod
    def cal_process(command_pipe, input_queue, output_queue, lock):
        """子进程计算函数"""
        while True:
            # 获取指令字
            command = command_pipe.recv()
            if command == AntColony.CommandType.Exit:
                # 结束进程
                break
            elif command == AntColony.CommandType.LoadDistance:
                # 读取目标函数矩阵和虚拟位移
                (distance_graph, virtual_distance) = input_queue.get()
            elif command == AntColony.CommandType.LoadVirtual:
                # 读取虚拟位移
                virtual_distance = input_queue.get()
            elif command == AntColony.CommandType.LoadPossible:
                # 读取概率矩阵
                possible_graph = input_queue.get()
            elif command == AntColony.CommandType.Calculate:
                # 开始计算
                while True:
                    try:
                        # 对于每个蚂蚁，调用search_path计算，并计算信息素增量矩阵
                        with lock:
                            ant = input_queue.get(timeout=0.01)
                        ant.search_path(distance_graph, possible_graph)
                        pheromone = np.zeros(np.shape(distance_graph), dtype=np.float64)
                        dp = 1.0 / (ant.total_distance - virtual_distance)

                        def add_pheromone(src, dest):
                            pheromone[src][dest] += dp

                        list(map(add_pheromone, ant.src_index, ant.dest_index))
                        with lock:
                            output_queue.put((ant, pheromone))
                    except multiprocessing.queues.Empty:
                        break
            else:
                pass
            # 完成指令后，做出回应
            command_pipe.send(True)

    def load_heuristic_graph(self, graph):
        """读入启发因子矩阵，并开启多进程"""
        super().load_heuristic_graph(graph)
        self.stop()
        # 创建控制子进程的指令管道，父进程输出指令，子进程返回回应
        pipes = [multiprocessing.Pipe(True) for _ in range(self._max_core)]
        self._process = [multiprocessing.Process(
                target=AntColony.cal_process,
                args=(pipe[0], self._task_queue, self._result_queue, self._queue_lock),
                daemon=True) for pipe in pipes]
        self._command_pipes = [pipe[1] for pipe in pipes]
        [p.start() for p in self._process]
        # 将目标函数矩阵和虚拟位移输入子进程
        [self._task_queue.put((self._distance_graph, self._virtual_distance)) for _ in self._process]
        [pipe.send(self.CommandType.LoadDistance) for pipe in self._command_pipes]
        _ = [pipe.recv() for pipe in self._command_pipes]

    def set_virtual_distance(self, virtual, particle_sum):
        super().set_virtual_distance(virtual, particle_sum)
        # 将计算所得虚拟输入子进程
        [self._task_queue.put(self._virtual_distance) for _ in self._process]
        [pipe.send(self.CommandType.LoadVirtual) for pipe in self._command_pipes]
        _ = [pipe.recv() for pipe in self._command_pipes]

    def stop(self):
        """停止多进程"""
        [p.terminate() for p in self._process]
        self._process = []
        [p.close() for p in self._command_pipes]
        self._command_pipes = []

    def calculate(self):
        """蚁群算法核心计算函数"""
        # 计算概率矩阵，并更新最优蚂蚁的目标函数和（由于反馈机制，不同迭代的_distance_graph可能不一致）
        self._update_possibility_graph()
        self.best_ant.cal_total_distance(self._distance_graph)
        self._random_src()
        # 将概率矩阵输入子进程
        [self._task_queue.put(self._possible_graph) for _ in self._process]
        [pipe.send(self.CommandType.LoadPossible) for pipe in self._command_pipes]
        _ = [pipe.recv() for pipe in self._command_pipes]
        # 将所有蚂蚁任务输入子进程，并开启计算
        [self._task_queue.put(ant) for ant in self.ants]
        [pipe.send(self.CommandType.Calculate) for pipe in self._command_pipes]

        pheromone = np.zeros((self._src_num, self._dest_num), dtype=np.float64)
        for i in range(self._ANT_NUM):
            # 读取计算完成的蚂蚁和对应的信息素矩阵增量
            (ant, p) = self._result_queue.get()
            self.ants[i] = ant
            pheromone += p
            # 与当前最优蚂蚁比较
            if ant.total_distance < self.best_ant.total_distance:
                # 更新最优解
                self.best_ant = copy.deepcopy(ant)

        _ = [pipe.recv() for pipe in self._command_pipes]
        # 更新信息素
        self._update_pheromone_graph(pheromone)

        return self.best_ant.src_index, self.best_ant.dest_index, self.best_ant.total_distance - self._virtual_distance
