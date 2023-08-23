"""
三维粒子生成算法，包含读入流场、标准流场和自制流场
"""

import enum
import pickle
import random
import re
import numpy as np
import scipy.io


class Functions(enum.Enum):
    """一个公共的接口，指明本文件中可调用的函数"""
    StdFlow = '无粒子缺失标准流场'
    MissingFlow = '粒子缺失流场'
    LabMadeFlow = '自制流场'


# 标准库选择
std_data_name = r'D:\PIV-STD_1999\351\PTC%03d.DAT'


class LoadFlow(object):
    """从MAT文件中读入一对粒子坐标

    类变量：
        PARAMETER_NAMES：保存了所需参数及对应描述（类常量），派生类应按需重写这个量

    成员变量：
        _particle_size：表述输出各帧粒子个数的元组
    """
    PARAMETER_NAMES = {
        'file_a': '前帧MAT文件路径',
        'file_b': '后帧MAT文件路径'
    }

    def __init__(self, parameters):
        """初始化类，将所需参数读入"""
        for name in self.PARAMETER_NAMES:
            setattr(self, '_'+name.upper(), parameters[name])
        self._particle_size = (0, 0)
        super().__init__()

    def __call__(self, *args, **kwargs):
        """定义调用方式，读入函数为self._generate"""
        particles = self._generate(*args, **kwargs)
        self._particle_size = (particles[0].size, particles[1].size)
        return particles

    def _generate(self):
        # 从path参数所指文件中读入双帧粒子坐标数据
        # 输入需为mat文件，并包含如下词条
        try:
            mat_a = scipy.io.loadmat(self._FILE_A)
            src_particles = np.hstack([np.atleast_2d(a).T for a in [mat_a['x'][0], mat_a['y'][0], mat_a['z']]])
        except IOError:
            print('无法打开文件:' + self._FILE_A)
            return (np.array([]), np.array([]))
        except ValueError:
            print(self._FILE_A + '不是有效的MATLAB数据文件')
            return (np.array([]), np.array([]))
        except KeyError as e:
            print('格式错误：' + self._FILE_A + '中找不到条目' + e.args[0])
            return (np.array([]), np.array([]))
        try:
            mat_b = scipy.io.loadmat(self._FILE_B)
            dest_particles = np.hstack([np.atleast_2d(a).T for a in [mat_b['x'][0], mat_b['y'][0], mat_b['z']]])
        except IOError:
            print('无法打开文件:' + self._FILE_B)
            return (np.array([]), np.array([]))
        except ValueError:
            print(self._FILE_B + '不是有效的MATLAB数据文件')
            return (np.array([]), np.array([]))
        except KeyError as e:
            print('格式错误：' + self._FILE_B + '中找不到条目' + e.args[0])
            return (np.array([]), np.array([]))
        # 无错误
        return (src_particles, dest_particles)


class ParticleGenerator3DBase(object):
    """三维粒子生成器的基类，定义了调用方式

    类变量：
        PARAMETER_NAMES：保存了所需参数及对应描述（类常量），派生类应按需重写这个量

    成员变量：
        _particle_size：表述输出各帧粒子个数的元组
    """
    PARAMETER_NAMES = {
        'num': '粒子数量',
        'record_generator': '记录生成器输出，用于调试',
        'lock_generator': '锁定生成器输出为record，用于调试'
    }

    def __init__(self, parameters):
        """初始化类，将所需参数读入"""
        for name in self.PARAMETER_NAMES:
            setattr(self, '_'+name.upper(), parameters[name])
        self._particle_size = (self._NUM, self._NUM)
        super().__init__()

    def __call__(self, *args, **kwargs):
        """定义调用方式，核心计算函数为self._generate"""
        # _LOCK_GENERATOR表示锁定输出，从文件中读取固定输出（用于调试计算段代码）
        try:
            if self._LOCK_GENERATOR is False:
                raise FileNotFoundError
            with open('particles.dat', 'rb') as fp:
                particles = pickle.load(fp)
            # 如果文件记录的输出粒子数量与参数给出的粒子数量不符，则重新生成并记录
            if tuple(len(frame) for frame in particles) != self._particle_size:
                self._RECORD_GENERATOR = True
                raise FileNotFoundError
        # 调用生成器生成全新粒子坐标
        except (FileNotFoundError, KeyError):
            particles = self._generate(*args, **kwargs)
        # 根据_RECORD_GENERATOR标记将当前输出记录到文件中，用于_LOCK_GENERATORD读取
        if self._RECORD_GENERATOR:
            with open('particles.dat', 'wb') as fp:
                pickle.dump(particles, fp)
        return particles

    def _generate(self):
        """生成粒子坐标，后续不同派生类应重载本函数"""
        return np.zeros((self._NUM, 3), dtype=np.float), np.zeros((self._NUM, 3), dtype=np.float)


class StdFlow(ParticleGenerator3DBase):
    """根据标准数据集生成流场

    成员变量：
        _start_frame：起始的数据帧序号，数据集目录及格式由全局变量std_data_name保存
    """
    PARAMETER_NAMES = ParticleGenerator3DBase.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['straddle_frames'] = '跨帧数量'

    def __init__(self, parameters):
        super().__init__(parameters)
        self._start_frame = 0

    def _generate(self):
        # 根据数据集目录以及跨帧数量，确定前帧后帧文件名
        path_a = std_data_name % self._start_frame
        path_b = std_data_name % (self._start_frame + self._STRADDLE_FRAMES)
        # 待输出前后帧粒子坐标
        src_particles = np.zeros((self._NUM, 3), dtype=np.float)
        dest_particles = np.zeros((self._NUM, 3), dtype=np.float)

        def pos_from_line(line_list, f1, f2):
            """根据相同粒子ID的粒子对所在文件的行号信息line_list，和两帧数据文件f1, f2，输出两帧粒子坐标
                粒子坐标保存在src_particles和dest_particles中，若ID不匹配，抛出ValueError
            """
            # 从所有相同ID的粒子对中随机选择_NUM数量的粒子输出
            out_list = random.sample(line_list, self._NUM)
            line1 = f1.readlines()
            line2 = f2.readlines()
            # 根据相同ID粒子对信息文件记录，找到两帧粒子数据的对应行，并读取坐标
            for i, out in enumerate(out_list):
                (p1, p2) = out
                m1 = re.match(r'(?:\s*)(\d+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)', line1[p1])
                src_particles[i] = [float(m1.group(j)) for j in range(2, 5)]
                m2 = re.match(r'(?:\s*)(\d+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)', line2[p2])
                dest_particles[i] = [float(m2.group(j)) for j in range(2, 5)]
                # 如果检查发现粒子ID不同，则粒子对文件有错误
                if int(m1.group(1)) != int(m2.group(1)):
                    raise ValueError

        with open(path_a, 'r') as f1, open(path_b, 'r') as f2:
            try:
                # 尝试从文件中读取缓存的相同ID粒子对信息
                with open('particlelist.dat', 'rb') as fp:
                    same_list = pickle.load(fp)
                pos_from_line(same_list, f1, f2)
            # 若导入粒子对信息失败或信息有错误
            except Exception:
                same_list = []
                f1.seek(0, 0)
                f2.seek(0, 0)
                line1 = f1.readlines()
                line2 = f2.readlines()
                l1 = len(line1)
                l2 = len(line2)
                p1 = 0
                p2 = 0
                # 对两个文件的所有粒子进行扫描，确定相同ID的粒子对
                while p1 < l1 and p2 < l2:
                    m1 = re.match(r'(?:\s*)(\d+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)', line1[p1])
                    n1 = int(m1.group(1))
                    m2 = re.match(r'(?:\s*)(\d+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)(?:\s*)([\d.-]+)', line2[p2])
                    n2 = int(m2.group(1))
                    if n1 > n2:
                        p2 += 1
                    elif n1 < n2:
                        p1 += 1
                    else:
                        same_list.append((p1, p2))
                        p1 += 1
                        p2 += 1
                # 将粒子对信息缓存到文件中，便于之后访问
                with open('particlelist.dat', 'wb') as fp:
                    pickle.dump(same_list, fp)
                f1.seek(0, 0)
                f2.seek(0, 0)
                pos_from_line(same_list, f1, f2)

        return src_particles, dest_particles


class MissingFlow(StdFlow):
    """根据标准数据集生成流场，考虑粒子缺失（缺失只发生在前帧）"""
    PARAMETER_NAMES = StdFlow.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['missing_rate'] = '粒子缺失率'
    PARAMETER_NAMES['missing_bias'] = '粒子缺失后帧偏置'

    def __init__(self, parameters):
        super().__init__(parameters)
        self._particle_size = (self._NUM-int(self._MISSING_RATE * self._NUM) + int(self._MISSING_RATE * self._MISSING_BIAS * self._NUM), self._NUM-int(self._MISSING_RATE * self._MISSING_BIAS * self._NUM))

    def _generate(self):
        # 根据标准流场生成粒子，将前帧粒子删除一部分作为缺失粒子
        src_particles, dest_particles = super()._generate()
        return np.vstack((src_particles[:self._NUM - int(self._MISSING_RATE*self._NUM)], src_particles[self._NUM - int(self._MISSING_RATE*self._MISSING_BIAS*self._NUM):])), \
            dest_particles[:self._NUM - int(self._MISSING_RATE*self._MISSING_BIAS*self._NUM)]


class LabMadeFlow(ParticleGenerator3DBase):
    """实验室自制流场，可自由控制比位移"""
    PARAMETER_NAMES = StdFlow.PARAMETER_NAMES.copy()
    PARAMETER_NAMES['delta'] = '比位移'

    def _generate(self):
        # 提供了三种不同流场函数可供选用
        return self.__generate1()

    def __generate1(self):
        """等强度剪切流叠加流向涡，流向涡以刚体旋转为模型"""
        # xmax为流向长度，r为涡半径，yabove为中心平面距流向速度为0的距离（距假想壁面的距离）
        xmax = 10.0
        r = 5.0
        yabove = 10
        # _DELTA为待生成流场的比位移要求，分母常数为校订常数，需根据实际流场尺寸以及强度关系实验模拟校正
        # 校正代码可选用主函数示例部分
        intensity = self._DELTA / 10.075
        dudy = intensity
        omega = intensity
        # 根据实际存在粒子的流场尺寸和粒子数量进一步校订，此步骤为自动校正
        invalid_percent = dudy * abs(yabove) / xmax if abs(yabove) > 2*r else dudy*((yabove+r)**2+(r-yabove)**2)/(4*r*xmax)
        d = ((1-invalid_percent) * 6*r*r*xmax / self._NUM)**(1/3)
        dudy *= d
        omega *= d

        # 随机生成若干粒子
        x1 = np.random.uniform(0.0, xmax, self._NUM)
        rho1 = np.random.uniform(0.0, r, self._NUM)
        theta1 = np.random.uniform(0.0, np.pi*2, self._NUM)
        y1 = rho1*np.cos(theta1)
        z1 = rho1*np.sin(theta1)
        # 根据理论模型生成第二帧粒子位置
        x2 = x1 + dudy * (y1+yabove)
        theta2 = theta1 + omega
        y2 = rho1 * np.cos(theta2)
        z2 = rho1 * np.sin(theta2)

        # 若粒子超出边界，则将出界粒子对随机平行迁移
        for i in range(self._NUM):
            if x2[i] > xmax:
                x = np.random.uniform(x2[i] - xmax, x1[i], 1)
                x1[i] = x1[i] - x
                x2[i] = x2[i] - x
            if x2[i] < 0:
                x = np.random.uniform(-x2[i], xmax - x1[i], 1)
                x1[i] = x1[i] + x
                x2[i] = x2[i] + x

        return np.hstack([np.atleast_2d(a).T for a in [x1, y1, z1]]), np.hstack([np.atleast_2d(a).T for a in [x2, y2, z2]])

    def __generate2(self):
        """等强度剪切流叠加流向涡，流向涡以兰姆-奥森涡（Lamb-Oseen）为模型"""
        # xmax为流向长度，r为涡半径，yabove为中心平面距流向速度为0的距离（距假想壁面的距离）
        xmax = 10.0
        r = 5.0
        yabove = 10
        # _DELTA为待生成流场的比位移要求，分母常数为校订常数，需根据实际流场尺寸以及强度关系实验模拟校正
        # 校正代码可选用主函数示例部分
        intensity = self._DELTA / 4.015
        dudy = intensity * 2/r
        gamma = intensity * r/6
        # 为防止出界，计算最远边界
        rhomax = np.sqrt((r**2 + np.sqrt(r**4 - 4*(gamma*(1 - np.exp(-r**2)))**2))/2)
        # 根据实际存在粒子的流场尺寸和粒子数量进一步校订，此步骤为自动校正
        invalid_percent = dudy * abs(yabove) / xmax if abs(yabove) > 2*r else dudy*((yabove+r)**2+(r-yabove)**2)/(4*r*xmax)
        invalid_percent *= (rhomax/r)**2
        d = ((1-invalid_percent) * 6*r*r*xmax / self._NUM)**(1/3)
        dudy *= d
        gamma *= d
        rhomax = np.sqrt((r**2 + np.sqrt(r**4 - 4*(gamma*(1 - np.exp(-r**2)))**2))/2)

        # 随机生成若干粒子
        x1 = np.random.uniform(0.0, xmax, self._NUM)
        rho1 = np.random.uniform(0.01, rhomax, self._NUM)
        theta1 = np.random.uniform(0.0, np.pi * 2, self._NUM)
        costheta = np.cos(theta1)
        sintheta = np.sin(theta1)
        y1 = rho1 * costheta
        z1 = rho1 * sintheta
        # 根据理论模型生成第二帧粒子位置
        x2 = x1 + dudy * (y1+yabove)
        vtheta = gamma / rho1 * (1 - np.exp(-rho1 ** 2))
        y2 = y1 - vtheta * sintheta
        z2 = z1 + vtheta * costheta

        # 若粒子超出边界，则将出界粒子对随机平行迁移
        for i in range(self._NUM):
            if x2[i] > xmax:
                x = np.random.uniform(x2[i] - xmax, x1[i], 1)
                x1[i] = x1[i] - x
                x2[i] = x2[i] - x
            if x2[i] < 0:
                x = np.random.uniform(-x2[i], xmax - x1[i], 1)
                x1[i] = x1[i] + x
                x2[i] = x2[i] + x

        return np.hstack([np.atleast_2d(a).T for a in [x1, y1, z1]]), np.hstack([np.atleast_2d(a).T for a in [x2, y2, z2]])

    def __generate3(self):
        """等强度剪切流叠加流向涡，流向涡以兰姆-奥森涡（Lamb-Oseen）为模型，切速度直接化为角速度"""
        # xmax为流向长度，r为涡半径，yabove为中心平面距流向速度为0的距离（距假想壁面的距离）
        xmax = 10.0
        r = 5.0
        yabove = 10.0
        # _DELTA为待生成流场的比位移要求，分母常数为校订常数，需根据实际流场尺寸以及强度关系实验模拟校正
        # 校正代码可选用主函数示例部分
        intensity = self._DELTA / 2.786
        dudy = intensity / r
        gamma = intensity * r
        # 根据实际存在粒子的流场尺寸和粒子数量进一步校订，此步骤为自动校正
        invalid_percent = dudy * abs(yabove) / xmax if abs(yabove) > 2*r else dudy*((yabove+r)**2+(r-yabove)**2)/(4*r*xmax)
        d = ((1-invalid_percent) * 6*r*r*xmax / self._NUM)**(1/3)
        dudy *= d
        gamma *= d

        # 随机生成若干粒子
        x1 = np.random.uniform(0.0, xmax, self._NUM)
        rho1 = np.random.uniform(0.01, r, self._NUM)
        theta1 = np.random.uniform(0.0, np.pi * 2, self._NUM)
        y1 = rho1 * np.cos(theta1)
        z1 = rho1 * np.sin(theta1)
        # 根据理论模型生成第二帧粒子位置
        x2 = x1 + dudy * (y1+yabove)
        vtheta = gamma / rho1 * (1 - np.exp(-rho1 ** 2))
        dtheta = vtheta / rho1
        theta2 = theta1 + dtheta
        y2 = rho1 * np.cos(theta2)
        z2 = rho1 * np.sin(theta2)

        # 若粒子超出边界，则将出界粒子对随机平行迁移
        for i in range(self._NUM):
            if x2[i] > xmax:
                x = np.random.uniform(x2[i] - xmax, x1[i], 1)
                x1[i] = x1[i] - x
                x2[i] = x2[i] - x
            if x2[i] < 0:
                x = np.random.uniform(-x2[i], xmax - x1[i], 1)
                x1[i] = x1[i] + x
                x2[i] = x2[i] + x

        return np.hstack([np.atleast_2d(a).T for a in [x1, y1, z1]]), np.hstack([np.atleast_2d(a).T for a in [x2, y2, z2]])


if __name__ == '__main__':
    """ 校订LabMadeFlow中的校准参数
    a = LabMadeFlow({'num': 1000000, 'record_generator': False, 'lock_generator': False, 'straddle_frames': 4, 'missing_rate': 0.1, 'delta': 1})
    for _ in range(10):
        src, dest = a()
        dis = np.linalg.norm(dest - src, ord=2, axis=1)
        # invalid需根据不同流场参数，从类内debug得到invalid_percent
        # d需根据流场尺寸和粒子个数填写
        # 打印值为比位移，需确定打印值与输入的delta尽量一致
        invalid = 0.1245244170264337
        d = ((1-invalid) * 6*5*5*10 / 1000000)**(1/3)
        print(np.mean(dis)/d)
    # """

    """ 简易测试流场的单次计算结果
    import PTV3D
    src_particles, dest_particles = LabMadeFlow({'num': 1000, 'record_generator': False, 'lock_generator': False, 'straddle_frames': 4, 'missing_rate': 0.1, 'delta': 1})()
    aa = PTV3D.PTV3D()
    srcs, dests = aa.once(src_particles, dest_particles)
    with open('res.txt', 'a') as fp:
        for d in dests:
            fp.write('%d\t' % d)
        fp.write('\n')
    # """

    """ 将特定的计算结果三维显示（已废弃）
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    '''
    with open('particle3d.dat', 'wb') as f:
        pickle.dump((x1, y1, z1, x2, y2, z2), f)
    # '''
    with open('particle3d.dat', 'rb') as f:
        x1, y1, z1, x2, y2, z2 = pickle.load(f)
    srcs = [753,157,839,566,347,336,291,455,643,245,572,412,620,734,876,207,358,610,471,698,295,90,219,443,527,892,653,792,360,989,449,500,879,49,255,684,674,865,237,132,458,668,986,438,685,75,800,640,537,361,659,446,585,867,656,359,983,900,490,869,145,209,866,962,779,925,107,356,39,411,63,581,847,300,312,16,68,910,507,681,932,804,571,426,582,10,378,25,686,472,921,706,965,504,424,253,533,739,722,395,119,793,957,114,637,784,840,180,517,510,655,730,325,332,198,707,289,914,127,966,211,352,555,369,233,532,535,271,801,524,539,737,257,131,137,404,425,909,299,158,594,599,150,632,826,163,481,38,196,502,761,280,697,882,241,382,703,463,586,782,503,935,370,947,465,496,142,148,745,558,712,990,485,26,78,366,853,875,877,974,825,923,74,562,961,937,405,136,931,531,430,809,407,977,870,601,297,541,608,732,153,123,943,499,110,236,217,954,6,408,726,627,60,85,554,459,664,331,718,573,930,284,850,835,134,975,592,294,743,673,952,770,429,574,917,846,445,689,813,415,488,183,679,819,751,567,362,439,250,199,772,536,568,122,883,646,76,393,491,775,461,794,290,277,363,385,915,849,292,480,285,941,215,399,190,728,654,234,868,704,99,367,403,40,569,345,936,995,80,205,692,77,845,758,912,678,316,848,886,442,206,834,630,514,431,0,319,660,200,769,512,444,696,596,182,565,767,759,170,901,181,477,944,523,991,330,354,823,669,173,231,248,187,970,498,65,479,939,242,218,818,861,342,534,304,755,178,560,121,777,185,42,606,644,281,365,230,953,451,890,322,902,550,799,843,35,747,224,140,44,388,88,907,651,302,14,329,719,837,156,611,773,473,807,720,725,851,612,738,267,23,576,894,201,389,638,934,419,483,225,2,188,903,955,33,261,551,12,57,896,618,174,29,276,62,343,495,888,997,521,101,105,259,897,338,229,515,70,584,723,647,91,595,822,635,341,66,274,320,226,317,79,437,780,904,791,45,396,676,240,614,9,964,776,152,844,973,575,453,417,885,494,368,956,51,221,613,144,661,911,24,265,176,124,1,268,948,373,841,665,58,994,189,522,856,377,631,428,327,164,47,628,889,744,887,392,335,916,623,600,272,334,852,827,307,436,530,293,129,155,525,814,808,197,548,92,493,64,116,750,812,920,34,394,159,486,420,106,795,478,891,736,262,401,244,872,22,929,149,714,223,100,824,228,741,4,933,748,959,757,120,971,18,113,789,949,717,67,603,288,556,179,619,175,598,694,871,318,125,591,269,432,13,191,740,213,967,972,344,811,658,162,252,422,32,452,340,353,313,469,398,778,609,108,402,146,409,616,705,984,763,985,31,348,501,43,545,662,511,945,251,433,296,115,167,756,454,97,987,583,702,315,400,648,727,884,98,691,860,633,474,559,339,203,381,561,729,470,988,55,516,357,383,278,328,629,50,227,457,969,5,821,220,816,349,520,15,996,41,976,141,84,716,746,542,528,475,11,484,593,519,708,492,279,540,118,386,563,765,194,456,441,104,165,700,671,128,731,636,305,913,724,462,878,421,497,69,863,905,86,177,978,505,46,103,642,993,781,254,53,699,27,815,543,168,924,992,687,908,605,626,423,564,287,946,766,680,169,624,326,410,308,858,21,3,273,256,333,951,639,602,690,820,796,139,418,109,306,667,906,672,829,351,82,464,663,622,427,553,641,184,578,468,764,790,434,963,721,832,768,161,836,186,893,919,387,192,30,881,212,172,324,828,397,715,311,8,859,72,590,552,980,787,899,126,93,526,376,17,151,238,634,435,677,831,487,926,854,309,489,133,466,928,314,874,833,688,476,518,36,440,210,509,83,460,81,513,774,87,709,762,406,243,275,355,683,166,264,138,263,999,979,28,154,283,682,785,577,371,208,617,695,950,895,286,880,810,450,374,249,61,898,735,171,557,710,842,372,803,19,375,214,380,625,817,942,102,649,448,48,321,579,754,447,202,193,346,364,195,701,266,830,615,379,786,570,760,862,111,927,222,670,873,89,350,752,54,37,298,52,246,323,938,650,482,59,258,235,675,922,384,771,546,588,666,998,604,940,645,73,806,547,749,117,143,467,303,416,958,621,960,529,802,414,982,390,7,96,391,56,282,580,918,135,798,711,968,713,270,805,607,838,855,783,538,981,95,232,130,742,589,112,239,549,310,301,413,71,337,506,20,864,657,204,652,733,788,693,857,147,216,94,247,597,544,587,508,260,160,797]
    dests = [753,157,839,566,347,336,291,455,643,245,572,412,620,734,876,207,358,610,471,698,295,90,219,443,527,892,653,792,360,989,449,500,879,49,255,684,674,865,237,132,458,668,986,438,685,75,800,640,537,361,659,446,585,867,656,359,983,900,490,869,145,209,866,962,973,925,107,356,39,411,63,581,847,300,312,16,68,910,507,681,932,804,571,426,582,10,378,25,686,472,921,706,965,504,424,253,533,739,722,395,119,793,957,114,637,784,840,180,517,510,655,730,325,332,198,707,289,914,127,966,211,352,555,369,233,532,535,271,801,524,539,737,257,131,137,404,425,909,299,158,594,599,150,632,826,163,481,38,196,502,529,280,697,882,241,382,703,463,586,782,128,935,370,947,465,496,142,148,745,558,712,990,485,26,78,366,853,875,934,974,825,923,74,562,346,937,405,136,931,531,430,809,407,977,870,601,297,541,608,732,153,123,943,499,110,236,217,954,6,408,726,627,60,85,554,459,664,331,718,573,930,284,850,835,134,975,592,294,743,673,952,770,429,574,917,846,445,689,813,415,488,183,679,819,751,567,362,439,250,199,772,536,568,122,883,646,76,393,491,775,461,794,290,277,363,385,915,849,292,480,285,941,215,399,190,728,654,234,868,704,99,367,403,40,569,345,936,995,80,205,692,77,845,758,912,678,316,848,886,442,206,834,630,514,431,0,319,660,200,769,512,444,696,596,182,565,767,759,170,901,181,477,944,523,991,330,354,823,669,173,231,248,187,970,498,225,479,939,242,218,818,861,342,534,304,755,178,560,121,777,185,42,606,644,281,365,230,953,451,890,322,902,550,799,843,35,747,224,140,44,388,88,907,651,302,14,329,719,837,156,611,773,473,807,720,725,56,612,738,267,23,576,894,201,389,638,59,419,483,877,2,188,903,955,33,261,551,12,57,896,618,174,29,276,62,343,495,888,997,521,101,105,259,897,338,229,515,70,584,723,647,91,595,822,635,341,66,274,320,226,557,79,437,780,978,791,45,396,676,240,614,9,964,776,152,844,154,575,453,417,885,494,368,956,51,221,613,144,661,911,24,317,176,124,1,268,948,373,841,665,58,994,189,522,856,377,631,428,327,164,47,628,889,744,887,392,335,916,623,600,272,334,852,827,307,436,530,904,129,194,525,814,808,197,548,92,493,64,116,750,812,711,34,394,159,486,420,106,795,478,891,736,262,401,244,872,22,929,149,714,223,100,824,228,741,4,933,748,959,757,120,971,18,113,789,949,717,67,603,288,556,179,619,175,598,694,871,318,125,591,269,432,13,191,740,213,967,972,344,811,658,162,252,422,32,452,340,353,313,469,398,778,609,108,402,146,409,616,705,984,763,985,31,348,501,43,545,662,511,945,251,433,296,115,167,756,454,97,987,583,702,315,400,648,727,884,98,691,860,633,474,559,339,203,381,561,729,470,988,55,516,357,383,278,328,629,50,227,457,969,5,821,220,816,349,520,15,996,41,976,141,84,716,746,542,528,475,11,484,593,519,708,492,279,862,118,386,563,765,20,456,441,104,165,700,671,779,731,636,305,913,724,462,878,421,497,69,863,905,86,177,961,505,46,103,642,993,781,254,53,699,27,815,543,168,924,992,687,908,605,626,423,564,287,946,766,680,169,624,326,410,308,858,21,3,273,256,333,951,639,602,690,820,796,139,418,109,306,667,906,672,829,351,82,464,663,622,427,553,641,184,578,468,764,540,434,963,721,832,768,161,836,186,893,919,387,210,30,881,212,172,324,828,397,715,311,8,859,72,590,552,980,787,899,126,93,526,376,17,151,238,634,435,677,831,487,926,854,309,489,133,466,928,314,874,833,688,476,518,36,440,752,509,83,460,81,513,774,87,709,762,406,243,275,355,683,166,264,138,263,999,979,28,851,283,682,785,577,371,208,617,695,950,895,286,880,810,450,374,249,61,898,790,171,652,710,842,372,803,19,375,214,380,625,817,942,102,649,448,48,321,579,754,447,202,193,192,364,195,701,266,830,615,379,786,570,760,293,111,927,222,670,873,89,350,506,54,37,298,52,246,323,938,650,482,735,258,235,675,922,384,771,546,588,666,998,604,940,645,73,806,547,749,117,143,467,303,416,958,621,960,503,802,414,982,390,7,96,391,761,282,580,918,135,798,920,968,713,270,805,607,838,855,783,538,981,95,232,130,742,589,112,239,549,310,301,413,71,337,657,155,864,65,204,597,733,788,693,857,147,216,94,247,265,544,587,508,260,160,797]
    xo1 = []
    xo2 = []
    yo1 = []
    yo2 = []
    zo1 = []
    zo2 = []
    right = []
    for i in range(len(srcs)):
        src = srcs[i]
        dest = dests[i]
        if src != dest:
            xo1.append(x1[src])
            xo2.append(x2[dest])
            yo1.append(y1[src])
            yo2.append(y2[dest])
            zo1.append(z1[src])
            zo2.append(z2[dest])
    xo1 = np.array(xo1)
    xo2 = np.array(xo2)
    yo1 = np.array(yo1)
    yo2 = np.array(yo2)
    zo1 = np.array(zo1)
    zo2 = np.array(zo2)
    aa = 0
    for i in range(len(xo1)):
        if xo1[i] > 0.8 or xo1[i] < -0.8 or xo2[i] < -0.8 or xo2[i] > 0.8 or yo1[i] > 0.6 or yo1[i] < -0.6 or yo2[i] < -0.6 or yo2[i] > 0.6 or zo1[i] > 0.2 or zo1[i] < -0.2 or zo2[i] < -0.2 or zo2[i] > 0.2:
            aa += 1
    figure2d = plt.figure()
    figure3d = Axes3D(figure2d)

    figure3d.scatter(xo1, yo1, zo1, c='k', s=10)
    figure3d.scatter(xo2, yo2, zo2, c='r', s=10)
    figure3d.quiver(xo1, yo1, zo1, xo2-xo1, yo2-yo1, zo2-zo1)
    '''
    figure3d.scatter(x1, y1, z1, c='k', s=5)
    figure3d.scatter(x2, y2, z2, c='r', s=5)
    '''
    figure3d.set_xlabel('x')
    figure3d.set_ylabel('y')
    figure3d.set_zlabel('z')
    plt.show()
    # """
