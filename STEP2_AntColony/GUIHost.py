"""
二维蚁群算法入口，GUI界面
"""

import importlib
import enum
import numpy as np
import scipy.io
import os
import sys
import threading
from functools import partial
from distutils.util import strtobool
import tkinter.filedialog
import tkinter.messagebox
import tkinter.ttk
import tkinter.constants


class Graph2DType(enum.Enum):
    Heuristic = '正常'
    Heuristic_C = 'C加速'


class SolverType(enum.Enum):
    Algorithm = '正常'
    Algorithm_Multi = '多进程'
    Algorithm_C = 'C加速'


class GUI(tkinter.Tk):
    """外观设计代码，逻辑处理在GUIHost子类中实现"""
    def __init__(self):
        super().__init__()
        self._create_menu()

        self.resizable(0, 0)
        self.iconbitmap(default='gui.ico')
        self.title("PTV蚁群算法")

    def _create_menu(self):
        """制作菜单栏"""
        self._menu = tkinter.Menu(self)
        menu_file = tkinter.Menu(self._menu, tearoff=False)
        menu_file.add_command(label='打开数据', command=self.load)
        menu_file.add_command(label='保存结果', command=self.save, state=tkinter.DISABLED)
        menu_file.add_separator()
        menu_file.add_command(label='退出', command=self.quit)
        self._menu.add_cascade(label='文件', menu=menu_file)
        menu_algorithm = tkinter.Menu(self._menu, tearoff=False)
        menu_algorithm.add_command(label='选择模块和函数', command=self.choose_module)
        menu_algorithm.add_command(label='设置参数', command=self.set_parameter)
        self._menu.add_cascade(label='参数', menu=menu_algorithm)
        menu_cal = tkinter.Menu(self._menu, tearoff=False)
        menu_cal.add_command(label='计算预热', command=self.start, state=tkinter.DISABLED)
        menu_cal.add_separator()
        menu_cal.add_command(label='开始计算', command=self.search_path, state=tkinter.DISABLED)
        menu_cal.add_command(label='计算一步', command=self.cal_once, state=tkinter.DISABLED)
        self._menu.add_cascade(label='计算', menu=menu_cal)

        self._refresh_menu()
        self['menu'] = self._menu

    def _refresh_menu(self):
        """将菜单布置为需要计算预热的模式（无法直接开启计算）"""
        cal_menu = self._menu.winfo_children()[self._menu.index('计算') - 1]
        try:
            cal_menu.entryconfig('继续计算', label='开始计算')
        except tkinter.TclError:
            pass
        try:
            cal_menu.entryconfig('暂停计算', label='开始计算')
        except tkinter.TclError:
            pass
        finally:
            cal_menu.entryconfig('开始计算', state=tkinter.DISABLED, command=self.search_path)
        try:
            cal_menu.entryconfig('重新计算', label='计算预热')
        except tkinter.TclError:
            pass
        finally:
            cal_menu.entryconfig('计算预热', state=tkinter.NORMAL)
        cal_menu.entryconfig('计算一步', state=tkinter.DISABLED)

    def _prepared_menu(self):
        """将菜单布置为可以开始计算的模式"""
        cal_menu = self._menu.winfo_children()[self._menu.index('计算') - 1]
        try:
            cal_menu.entryconfig('继续计算', label='开始计算')
        except tkinter.TclError:
            pass
        try:
            cal_menu.entryconfig('暂停计算', label='开始计算')
        except tkinter.TclError:
            pass
        finally:
            cal_menu.entryconfig('开始计算', state=tkinter.NORMAL, command=self.search_path)
        try:
            cal_menu.entryconfig('计算预热', label='重新计算')
        except tkinter.TclError:
            pass
        cal_menu.entryconfig('计算一步', state=tkinter.NORMAL)
        file_menu = self._menu.winfo_children()[self._menu.index('文件') - 1]
        file_menu.entryconfig('保存结果', state=tkinter.DISABLED)

    def _running_menu(self):
        """已经开始计算时的菜单模式"""
        cal_menu = self._menu.winfo_children()[self._menu.index('计算') - 1]
        cal_menu.entryconfig('重新计算', state=tkinter.DISABLED)
        try:
            cal_menu.entryconfig('开始计算', label='暂停计算')
        except tkinter.TclError:
            pass
        try:
            cal_menu.entryconfig('继续计算', label='暂停计算')
        except tkinter.TclError:
            pass
        finally:
            cal_menu.entryconfig('暂停计算', state=tkinter.NORMAL, command=self.pause)
        cal_menu.entryconfig('计算一步', state=tkinter.DISABLED)
        file_menu = self._menu.winfo_children()[self._menu.index('文件') - 1]
        file_menu.entryconfig('保存结果', state=tkinter.NORMAL)

    def _pause_menu(self):
        """暂停时的菜单模式"""
        cal_menu = self._menu.winfo_children()[self._menu.index('计算') - 1]
        try:
            cal_menu.entryconfig('暂停计算', label='继续计算', command=self.search_path)
        except tkinter.TclError:
            pass
        cal_menu.entryconfig('重新计算', state=tkinter.NORMAL)
        cal_menu.entryconfig('计算一步', state=tkinter.NORMAL)


class GUIHost(GUI):
    """实现GUI逻辑和渲染，调用计算函数

    成员变量：
        _r：绘图所用表示粒子的圆形的半径
        _lock, _running：组合控制算法的启停
        _once：用于控制计算一步的标记
        _max_size：本程序最大画布大小
        _scale：记录显示坐标与输入粒子坐标的缩放比
        _hello_image, _hello_canvas：导入的欢迎界面图像以及对应的画布
        _canvas：当前显示的画布

        _module_selection, _parameter：用于获取对话框的返回值
        _algorithm_module_list, _heuristic_module_list可供导入的所有模块
        _algorithm_module, _heuristic_module：选择导入的模块
        _module_selection：包含与ChooseModule交互的数据，类型为字典
        _algorithm_module_available, _heuristic_module_available：可用的模块名列表
        _algorithm_function_available, _heuristic_function_available：模块内可用的函数名字典，键名为模块名，键值为可用的函数枚举值列表
        _algorithm_module, _heuristic_module, _algorithm_function, _heuristic_function：目前选择的模块名与函数名
        _parameter, _parameter_label：所选algorithm和heuristic所需参数与标签，与SetParameter交互
    """
    def __init__(self):
        super().__init__()

        self._r = 5
        self._lock = threading.RLock()
        self._running = False
        self._once = False

        self._module_selection = {}
        self._parameter = {}
        self._parameter_label = {}
        self._algorithm_module_list = []
        self._heuristic_module_list = []
        self._module_selection['algorithm_module_available'] = []
        self._module_selection['heuristic_module_available'] = []
        self._module_selection['algorithm_function_available'] = {}
        self._module_selection['heuristic_function_available'] = {}

        def judge_available(function_dict, module_list, name_list, name):
            """判断模块名name是否可导入，如果可以则把模块内的可用函数放入function_dict中，模块放入module_list中，名称放入name_list中"""
            try:
                module = importlib.import_module(name)
                function_dict[name] = list(module.Functions)
                module_list.append(module)
                name_list.append(name)
            except Exception:
                pass

        # 判断并初始化available词条
        [judge_available(self._module_selection['algorithm_function_available'], self._algorithm_module_list, self._module_selection['algorithm_module_available'], name) for name in SolverType.__members__]
        [judge_available(self._module_selection['heuristic_function_available'], self._heuristic_module_list, self._module_selection['heuristic_module_available'], name) for name in Graph2DType.__members__]
        # 选择默认模块与函数，及对应的参数
        self._module_selection['algorithm_module'] = self._module_selection['algorithm_module_available'][-1]
        self._module_selection['heuristic_module'] = self._module_selection['heuristic_module_available'][-1]
        self._module_selection['algorithm_function'] = self._module_selection['algorithm_function_available'][self._module_selection['algorithm_module']][-1].name
        self._module_selection['heuristic_function'] = self._module_selection['heuristic_function_available'][self._module_selection['heuristic_module']][-1].name
        self._algorithm_module = self._algorithm_module_list[-1]
        self._heuristic_module = self._heuristic_module_list[-1]
        self._parameter = self.load_default_parameter()

        # 确定最大画布以及缩放比
        self._max_size = (1000, 1000)
        self._scale = 1
        # 绘制欢迎界面
        self._hello_image = tkinter.PhotoImage(file='hello.gif')
        self._hello_canvas = tkinter.Canvas(
            self,
            width=self._hello_image.width(),
            height=self._hello_image.height(),
        )
        self._hello_canvas.create_image(self._hello_image.width() // 2, self._hello_image.height() // 2, image=self._hello_image)
        self._canvas = self._hello_canvas
        self._show_hello_image()

    def load(self, path=''):
        """读入粒子坐标文件，path若定义则直接使用，不用GUI选择"""
        # 停止线程
        with self._lock:
            self._running = False
        self._refresh_menu()

        # 获得路径，如果路径不合法，或者数据不满足要求，回到欢迎界面
        if path.strip() == '':
            path = tkinter.filedialog.askopenfilename(initialdir=sys.path[0])
        if path.strip() == '':
            return
        try:
            # 输入需为mat文件，并包含如下词条
            mat_in = scipy.io.loadmat(path)
            self.src_particles = np.hstack([np.atleast_2d(a).T for a in [mat_in['coordi1'][0], mat_in['coordj1'][0]]])
            self.dest_particles = np.hstack([np.atleast_2d(a).T for a in [mat_in['coordi2'][0], mat_in['coordj2'][0]]])
            (self.src_particle_num, self.dest_particle_num) = (len(self.src_particles), len(self.dest_particles))
        except IOError:
            self._show_hello_image()
            tkinter.messagebox.showerror('文件错误', '无法打开文件:' + path)
            return
        except ValueError:
            self._show_hello_image()
            tkinter.messagebox.showerror('格式错误', path + '不是有效的MATLAB数据文件')
            return
        except KeyError as e:
            self._show_hello_image()
            tkinter.messagebox.showerror('文件错误', '格式错误：' + path + '中找不到条目' + e.args[0])
            return
        # 若无错误，则进入粒子渲染阶段
        self._show_particle()

    def _show_hello_image(self):
        """重新载入欢迎界面"""
        self._canvas.pack_forget()
        self._scale = 1
        self._canvas = self._hello_canvas
        self._canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)

    def _show_particle(self):
        """绘制粒子"""
        self._canvas.pack_forget()
        # 按照数据等比例缩放，保证某边与最大边界长度一致，且另一边小于等于最大长度，2为调整屏幕坐标常数
        min_pos = np.minimum(np.min(self.src_particles, axis=0), np.min(self.dest_particles, axis=0))
        max_pos = np.maximum(np.max(self.src_particles, axis=0), np.max(self.dest_particles, axis=0))
        size = max_pos - min_pos
        self._scale = min((np.array(self._max_size)-2*self._r-2)/(max_pos-min_pos))
        self._canvas = tkinter.Canvas(
            self,
            width=size[0]*self._scale+2*self._r+2,
            height=size[1]*self._scale+2*self._r+2,
            bg="#EBEBEB",  # 背景灰色
        )
        self._canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)

        # points为坐标，nodes为半径_r的圆形节点对象
        self._src_points = (self.src_particles-min_pos)*self._scale + self._r + 2
        self._dest_points = (self.dest_particles-min_pos)*self._scale + self._r + 2
        self._src_nodes = [self._canvas.create_oval(
            p[0] - self._r, p[1] - self._r, p[0] + self._r, p[1] + self._r,
            fill="#ffffff",  # 填充白色
            outline="#000000",  # 轮廓黑色
            tags="node") for p in self._src_points]
        self._dest_nodes = [self._canvas.create_oval(
            p[0] - self._r, p[1] - self._r, p[0] + self._r, p[1] + self._r,
            fill="#ff0000",  # 填充红色
            outline="#000000",  # 轮廓黑色
            tags="node") for p in self._dest_points]

        '''显示坐标（已废除）
        [self._canvas.create_text(
            p[0], p[1]-(self._r+5),
            text='(%.2f, %.2f)' % (p[0], p[1]),
            fill='black') for p in self._src_points]
        [self._canvas.create_text(
            p[0], p[1]+(self._r+5),
            text='(%.2f, %.2f)' % (p[0], p[1]),
            fill='black') for p in self._dest_points]
        '''

    def save(self, path=''):
        """输出粒子配对关系，path若定义则直接使用，不用GUI选择"""
        # 停止线程
        with self._lock:
            self._running = False
        self._pause_menu()

        # 获得路径
        if path.strip() == '':
            path = tkinter.filedialog.asksaveasfilename(initialdir=sys.path[0], filetypes=[('Matlab Data', '.mat')])
        if path.strip() == '':
            return
        name, suffix = os.path.splitext(path)
        if suffix == '':
            path = name + '.mat'
        scipy.io.savemat(path, {'src': self._src_index+1, 'dest': self._dest_index+1, 'distance': self._distance})

    def quit(self):
        """退出程序"""
        with self._lock:
            self._running = False
        print("\n程序已退出...")
        super().quit()

    def start(self):
        """定义蚁群算法，并先行计算目标函数矩阵Heuristic"""
        # 初始化目标函数矩阵，并添加虚拟粒子
        try:
            self._graph = getattr(self._heuristic_module, self._module_selection['heuristic_function'])((self.src_particles, self.dest_particles), self._parameter)
            self._graph.add_margin()
        except AttributeError:
            tkinter.messagebox.showerror('数据错误', '请读取合适的数据')
            return
        except KeyError:
            tkinter.messagebox.showerror('算法错误', '请在运行之前先选择算法')
            return
        # 初始化蚁群算法
        try:
            self._aco = getattr(self._algorithm_module, self._module_selection['algorithm_function'])((self.src_particle_num, self.dest_particle_num), self._parameter)
            self._aco.load_heuristic_graph(self._graph.get_heuristic_graph())
            # 前后粒子不同，计算虚拟粒子目标函数
            virtual = np.mean(self._graph.get_heuristic_graph())
            self._graph.set_margin(virtual)
        except KeyError:
            tkinter.messagebox.showerror('算法错误', '请在运行之前先选择算法')
            return
        except AttributeError:
            tkinter.messagebox.showerror('算法错误', '请在运行之前先选择算法')
            return

        self._iter = 0
        self.title("PTV蚁群算法 迭代次数: 0")
        self._prepared_menu()

    def _init_module(self):
        """初始化模块，若不可用则恢复为可用列表的最后项"""
        try:
            self._algorithm_module = self._algorithm_module_list[self._module_selection['algorithm_module_available'].index(self._module_selection['algorithm_module'])]
        except ValueError:
            tkinter.messagebox.showerror('错误', '模块名'+SolverType[self._module_selection['algorithm_module']].value+'不可导入，已恢复为'+self._module_selection['algorithm_function_available'][-1])
            self._module_selection['algorithm_module'] = self._module_selection['algorithm_function_available'][-1]
            self._algorithm_module = self._algorithm_module_list[-1]
        try:
            self._heuristic_module = self._heuristic_module_list[self._module_selection['heuristic_module_available'].index(self._module_selection['heuristic_module'])]
        except ValueError:
            tkinter.messagebox.showerror('错误', '模块名'+Graph2DType[self._module_selection['heuristic_module']].value+'不可导入，已恢复为'+self._module_selection['heuristic_function_available'][-1])
            self._module_selection['heuristic_module'] = self._module_selection['heuristic_function_available'][-1]
            self._heuristic_module = self._heuristic_module_list[-1]

        algorithm_available_name = [value.name for value in self._module_selection['algorithm_function_available'][self._module_selection['algorithm_module']]]
        if self._module_selection['algorithm_function'] not in algorithm_available_name:
            self._module_selection['algorithm_function'] = algorithm_available_name[-1]
            tkinter.messagebox.showerror(
                '错误',
                '目标函数' + self._module_selection['algorithm_function'] + '不存在于' + SolverType[self._module_selection['algorithm_module']].value+'中，'
                + '已恢复为目标函数'+self._module_selection['algorithm_function_available'][self._module_selection['algorithm_module']][-1].value)
        heuristic_available_name = [value.name for value in self._module_selection['heuristic_function_available'][self._module_selection['heuristic_module']]]
        if self._module_selection['heuristic_function'] not in heuristic_available_name:
            self._module_selection['heuristic_function'] = heuristic_available_name[-1]
            tkinter.messagebox.showerror(
                '错误',
                '目标函数' + self._module_selection['heuristic_function'] + '不存在于' + Graph2DType[self._module_selection['heuristic_module']].value+'中，'
                + '已恢复为目标函数'+self._module_selection['heuristic_function_available'][self._module_selection['heuristic_module']][-1].value)

    def search_path(self):
        """创建线程，开始计算"""
        # _running用于控制计算线程，True表示开始，False表示计算停止
        with self._lock:
            self._running = True
        self._running_menu()

        cal_thread = threading.Thread(target=self.run, daemon=True)
        cal_thread.start()

    def run(self):
        """计算线程核心函数"""
        # _running为线程活动标记
        while self._running:
            self._iter += 1

            # 遍历每一只蚂蚁
            [self._src_index, self._dest_index, self._distance] = self._aco.calculate()
            true_particle = (self._src_index < self.src_particle_num) & (self._dest_index < self.dest_particle_num)
            self._src_index = self._src_index[true_particle]
            self._dest_index = self._dest_index[true_particle]

            virtual = 2 * self._distance / max(self.src_particle_num, self.dest_particle_num)
            self._graph.set_margin(virtual)

            self._canvas.delete("line")

            # 渲染配对关系（直线）
            list(map(lambda p0, p1: self._canvas.create_line(tuple(p0), tuple(p1), fill="#000000", tags="line"), self._src_points[self._src_index], self._dest_points[self._dest_index]))

            # 设置标题
            self.title("PTV蚁群算法 迭代次数: %d" % self._iter)
            print("迭代次数：", self._iter, u"最佳路径总距离：%.2f" % self._distance)
            # 更新画布
            self._canvas.update()

            # 只计算一次
            if self._once is True:
                with self._lock:
                    self._once = False
                self.pause()

    def pause(self):
        """停止计算线程"""
        with self._lock:
            self._running = False
        self._pause_menu()

    def cal_once(self):
        """设定计算线程为计算一次"""
        with self._lock:
            self._running = True
            self._once = True
        self._running_menu()

        cal_thread = threading.Thread(target=self.run)
        cal_thread.start()

    def choose_module(self):
        """打开选择模块窗口"""
        if self._running is True:
            self.pause()
        old_selection = self._module_selection.copy()
        choice_window = ChooseModule(self, self._module_selection)
        choice_window.focus_set()
        choice_window.grab_set()
        self.wait_window(choice_window)

        if self._module_selection != old_selection:
            self._init_module()
            self._refresh_menu()
        # 如果方法有变动，则载入默认参数
        algorithm_flag = self._module_selection['algorithm_function'] != old_selection['algorithm_function']
        heuristic_flag = self._module_selection['heuristic_function'] != old_selection['heuristic_function']
        if algorithm_flag or heuristic_flag:
            self._parameter = self.load_default_parameter(algorithm_flag, heuristic_flag)
            self._refresh_menu()
            self.set_parameter()

    def set_parameter(self):
        """打开设置参数窗口"""
        if self._running is True:
            self.pause()
        old_parameter = self._parameter.copy()
        choice_window = SetParameter(self, self._parameter, self._parameter_label)
        choice_window.focus_set()
        choice_window.grab_set()
        self.wait_window(choice_window)
        if self._parameter != old_parameter:
            self._refresh_menu()

    def load_default_parameter(self, algorithm_flag=True, heuristic_flag=True):
        """重新载入默认参数"""
        default = {
            'alpha': 1.0,
            'beta': 2.0,
            'beta2': 3.0,
            'rho': 0.5,
            'q': -1,
            'ant_num': 50,
            'random_src': True,
            'step': 360,
            'min_distance': 0.01,
            'min_similarity': 0.01,
            'max_neighbor': 5
        }
        if self._module_selection['heuristic_function'] == 'VoronoiGraph':
            default['beta'] = 6.0
        if self._module_selection['heuristic_function'] == 'RelaxGraph' or self._module_selection['heuristic_function'] == 'RelaxGraph':
            default['beta'] = 5.0

        # 重置所有所需的参数以及对应的标签
        try:
            parameters = {name: default[name] if algorithm_flag else self._parameter[name] for name in getattr(self._algorithm_module, self._module_selection['algorithm_function']).PARAMETER_NAMES}
            parameters.update({name: default[name] if heuristic_flag else self._parameter[name] for name in getattr(self._heuristic_module, self._module_selection['heuristic_function']).PARAMETER_NAMES})
        except KeyError:
            parameters = {name: default[name] for name in getattr(self._algorithm_module, self._module_selection['algorithm_function']).PARAMETER_NAMES}
            parameters.update({name: default[name] for name in getattr(self._heuristic_module, self._module_selection['heuristic_function']).PARAMETER_NAMES})
        self._parameter_label = getattr(self._algorithm_module, self._module_selection['algorithm_function']).PARAMETER_NAMES.copy()
        self._parameter_label.update(getattr(self._heuristic_module, self._module_selection['heuristic_function']).PARAMETER_NAMES)
        return parameters


class ChooseModuleUI(tkinter.Toplevel):
    """这个类仅实现界面生成功能，具体事件处理代码在子类ChooseModule中"""
    def __init__(self, master=None):
        super().__init__(master)
        self.title('选择模块')
        self.resizable(0, 0)
        self.create_widgets()

    def create_widgets(self):
        """创建控件"""
        # 为各种控件设置字体
        self._style = tkinter.ttk.Style()
        font_style = partial(self._style.configure, font=('宋体', 12))
        font_style('TLabelframe.Label')
        font_style('TButton')
        font_style('TRadiobutton')
        font_style('TLabel')
        frame_style = partial(tkinter.ttk.LabelFrame, self, style='TLabelframe')
        label_style = partial(tkinter.ttk.Label, self, style='TLabel')
        frame_row = 0
        # _disable_flag为是否有无法导入模块的标记
        self._disable_flag = False

        # 每个frame包含一个LabelFrame控制的选项组，其中各个选项由_create_radiobutton创建
        self._frame_algorithm_module = frame_style(text='蚁群算法模块')
        self._frame_algorithm_module.grid(row=frame_row, sticky=tkinter.constants.W, padx=5, pady=5)
        self._algorithm_module_var = tkinter.StringVar(value=self.modules['algorithm_module'])
        self._algorithm_module = [self._create_radiobutton(
            self._frame_algorithm_module, self._algorithm_module_var, self.modules['algorithm_module_available'],
            algorithm, index, command=self._create_functions_choices) for index, algorithm in enumerate(SolverType)]
        frame_row += 1

        self._frame_heuristic_module = frame_style(text='目标函数模块')
        self._frame_heuristic_module.grid(row=frame_row, sticky=tkinter.constants.W, padx=5, pady=5)
        self._heuristic_module_var = tkinter.StringVar(value=self.modules['heuristic_module'])
        self._heuristic_module = [self._create_radiobutton(
            self._frame_heuristic_module, self._heuristic_module_var, self.modules['heuristic_module_available'],
            heuristic, index, command=self._create_functions_choices) for index, heuristic in enumerate(Graph2DType)]
        frame_row += 1

        # 由于可选函数可能动态更新，故统一由_create_functions_choices控制生成下列frame中的选项
        self._frame_algorithm_function = frame_style(text='算法选择')
        self._frame_algorithm_function.grid(row=frame_row, sticky=tkinter.constants.W, padx=5, pady=5)
        self._algorithm_function_var = tkinter.StringVar(value=self.modules['algorithm_function'])
        self._algorithm_function = []
        frame_row += 1

        self._frame_heuristic_function = frame_style(text='目标函数选择')
        self._frame_heuristic_function.grid(row=frame_row, sticky=tkinter.constants.W, padx=5, pady=5)
        self._heuristic_function_var = tkinter.StringVar(value=self.modules['heuristic_function'])
        self._heuristic_function = []
        frame_row += 1

        self._create_functions_choices()

        # 若存在不可选选项，则添加一段说明
        if self._disable_flag:
            frame_row += 1
            self._annotation = label_style(text='注：灰色为导入失败或不支持的选项')
            self._annotation.grid(row=frame_row, pady=5)

        # 将两个按钮打包成frame
        self._button_frame = tkinter.ttk.Frame(self, style='TFrame')
        button_style = partial(tkinter.ttk.Button, self._button_frame, style='TButton')
        self._button_ok = button_style(text='确定', command=self.button_ok_cmd)
        self._button_ok.grid(row=0, column=0, padx=20, pady=20)
        self._button_cancel = button_style(text='取消', command=self.button_cancel_cmd)
        self._button_cancel.grid(row=0, column=1, padx=20, pady=20)
        self._button_frame.grid(row=frame_row)
        frame_row += 1

        self.bind('<Return>', self.button_ok_cmd)
        self.bind('<Escape>', self.button_cancel_cmd)
        self.protocol("WM_DELETE_WINDOW", self.button_cancel_cmd)

    def _create_functions_choices(self):
        """填充选择函数框架中的选项"""
        # available为从父窗口导入的数据，类型为list，每个元素为可用的函数名称的枚举值
        algorithm_module = self._algorithm_module_var.get()
        algorithm_available = self.modules['algorithm_function_available'][algorithm_module]
        algorithm_available_name = [value.name for value in algorithm_available]
        if self._algorithm_function_var.get() not in algorithm_available_name:
            self._algorithm_function_var.set(algorithm_available_name[-1])
        for widget in self._algorithm_function:
            widget.destroy()
        self._algorithm_function = [self._create_radiobutton(self._frame_algorithm_function, self._algorithm_function_var, algorithm_available_name, algorithm, index) for index, algorithm in enumerate(algorithm_available)]

        heuristic_module = self._heuristic_module_var.get()
        heuristic_available = self.modules['heuristic_function_available'][heuristic_module]
        heuristic_available_name = [value.name for value in heuristic_available]
        if self._heuristic_function_var.get() not in heuristic_available_name:
            self._heuristic_function_var.set(heuristic_available_name[-1])
        for widget in self._heuristic_function:
            widget.destroy()
        self._heuristic_function = [self._create_radiobutton(self._frame_heuristic_function, self._heuristic_function_var, heuristic_available_name, heuristic, index) for index, heuristic in enumerate(heuristic_available)]

    def _create_radiobutton(self, master, variable, available_list, value, index, command=None):
        """生成选项，并判断其是否可用"""
        # 一行包含三个选项
        button = tkinter.ttk.Radiobutton(master, text=value.value, value=value.name, variable=variable, style='TRadiobutton', command=command)
        button.grid(row=(index // 3), column=(index % 3), sticky=tkinter.constants.W, padx=20, pady=10)
        if value.name not in available_list:
            button.configure(state=tkinter.constants.DISABLED)
            self._disable_flag = True
        return button


class ChooseModule(ChooseModuleUI):
    """这个类实现具体的事件处理回调函数，界面生成代码在ChooseModuleUI中

    成员变量：
        modules：字典对象，父窗口导入的选项信息
    """
    def __init__(self, master, modules):
        self.modules = modules
        super().__init__(master)

    def button_ok_cmd(self, event=None):
        """处理按下确定键函数"""
        self.modules['algorithm_module'] = self._algorithm_module_var.get()
        self.modules['heuristic_module'] = self._heuristic_module_var.get()
        self.modules['algorithm_function'] = self._algorithm_function_var.get()
        self.modules['heuristic_function'] = self._heuristic_function_var.get()
        self.destroy()

    def button_cancel_cmd(self, event=None):
        """处理按下取消键函数"""
        self.destroy()


class SetParameterUI(tkinter.Toplevel):
    """这个类仅实现界面生成功能，具体事件处理代码在子类SetParameter中"""
    def __init__(self, master=None):
        super().__init__(master)
        self.title('修改参数')
        self.resizable(0, 0)
        self.create_widgets()

    def create_widgets(self):
        """创建控件"""
        # 为各种控件设置字体
        self._style = tkinter.ttk.Style()
        font_style = partial(self._style.configure, font=('宋体', 12))
        font_style('TButton')
        font_style('TLabel', anchor='e')
        font_style('TEntry')

        label_style = partial(tkinter.ttk.Label, self, style='TLabel')
        entry_style = partial(tkinter.ttk.Entry, self, style='TEntry')
        frame_row = 0

        def create_row(name, row):
            """创建参数输入行，一行由左侧的label说明和右侧的entry输入构成"""
            var = tkinter.StringVar(value=str(self.parameters[name]))
            label = label_style(text=self.labels[name])
            entry = entry_style(textvariable=var)
            label.grid(row=row, column=0, sticky=tkinter.constants.E, padx=20, pady=10)
            entry.grid(row=row, column=1, sticky=tkinter.constants.W, padx=20, pady=10)
            return (name, var, label, entry)

        # 创建用于参数输入的各行
        # 每个列表为一个四元组，分别代表参数变量名，参数输入变量，左侧label对象，右侧entry对象
        self._widgets = [create_row(name, frame_row+index) for index, name in enumerate(self.parameters)]
        frame_row += len(self.parameters)

        self._button_frame = tkinter.ttk.Frame(self, style='TFrame')
        button_style = partial(tkinter.ttk.Button, self._button_frame, style='TButton')
        self._button_default = button_style(text='默认', command=self.button_default_cmd)
        self._button_default.grid(row=0, column=0, padx=20, pady=20)
        self._button_ok = button_style(text='确定', command=self.button_ok_cmd)
        self._button_ok.grid(row=0, column=1, padx=20, pady=20)
        self._button_cancel = button_style(text='取消', command=self.button_cancel_cmd)
        self._button_cancel.grid(row=0, column=2, padx=20, pady=20)
        self._button_frame.grid(row=frame_row, column=0, columnspan=2)
        frame_row += 1

        self.bind('<Return>', self.button_ok_cmd)
        self.bind('<Escape>', self.button_cancel_cmd)
        self.protocol("WM_DELETE_WINDOW", self.button_cancel_cmd)


class SetParameter(SetParameterUI):
    """这个类实现具体的事件处理回调函数，界面生成代码在SetParameterUI中

    成员变量：
        parameters：字典对象，参数变量值
        labels：字典对象，参数对应的描述
        """
    def __init__(self, master, parameters, labels):
        self.parameters = parameters
        self.labels = labels
        super().__init__(master)

    def button_default_cmd(self, event=None):
        """处理按下默认键函数"""
        default = self.master.load_default_parameter()
        [row[1].set(str(default[row[0]])) for row in self._widgets]

    def button_ok_cmd(self, event=None):
        """处理按下确定键函数"""
        def adjust_type(string, value_type):
            """根据原变量的不同类型，对输入类型转换"""
            if value_type is int:
                value = float(string)
            elif value_type is bool:
                value = strtobool(string)
            else:
                value = string
            return value_type(value)
        try:
            self.parameters.update({row[0]: adjust_type(row[1].get(), type(self.parameters[row[0]])) for row in self._widgets})
        except ValueError:
            tkinter.messagebox.showerror('参数错误', '参数填写错误，请检查')
            return
        self.destroy()

    def button_cancel_cmd(self, event=None):
        """处理按下取消键函数"""
        self.destroy()


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
    GUIHost().mainloop()
