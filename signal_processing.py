import numpy as np
import itertools

class SignalSequence:
    """
    一个用于表示数字信号序列 x[n] 的类。
    
    该类使用一个内部 NumPy 数组来存储数据，并使用一个 `n0` 变量
    来标记数组中第一个元素对应的索引 n。
    
    满足实验要求 2:
    - 2a) 长度有限 (由输入数据决定)
    - 2b) 任意起始位置 (通过 `n0` 参数)
    - 2c) 任意位置读写 (通过 `__getitem__` 和 `__setitem__`)
    """

    def __init__(self, data, n0=0):
        """
        初始化信号序列
        
        参数:
        - data (list 或 np.array): 包含序列非零值的列表或数组。
        - n0 (int): 'data' 列表中第一个元素对应的序列索引 n。
                      例如: x = [1, 2, 3], n0 = -1
                      表示 x[-1]=1, x[0]=2, x[1]=3
        """
        # 统一使用 NumPy 数组进行高效计算
        self.data = np.asarray(data)
        # 序列 data[0] 对应的索引 n
        self.n0 = int(n0)

    @property
    def n_start(self):
        """序列的起始索引 n"""
        return self.n0
        
    @property
    def n_end(self):
        """序列的结束索引 n"""
        return self.n0 + len(self.data) - 1

    @property
    def indices(self):
        """返回一个包含所有n索引的 NumPy 数组"""
        return np.arange(self.n_start, self.n_end + 1)

    def __getitem__(self, n):
        """
        实现任意位置读取 (要求 2c: 读)
        例如: y = x[n]
        """
        # 1. 计算 n 对应的内部数组索引
        internal_idx = n - self.n0
        
        # 2. 检查索引是否在有效范围内
        if 0 <= internal_idx < len(self.data):
            return self.data[internal_idx]
        else:
            # 3. 如果在范围之外，根据 DSP 惯例返回 0
            return 0

    def __setitem__(self, n, value):
        """
        实现任意位置写入 (要求 2c: 写)
        例如: x[n] = 5
        
        如果 n 在当前范围之外，会自动扩展序列并补零。
        """
        internal_idx = n - self.n0
        
        # --- 情况 1: 写入位置在当前范围之前 ---
        if internal_idx < 0:
            # 需要在 data 数组的 *前面* 补零
            num_prepend_zeros = -internal_idx
            # 使用 np.pad 在前面补零
            self.data = np.pad(self.data, (num_prepend_zeros, 0), 'constant')
            # 更新新的起始索引
            self.n0 = n
            # 写入值 (现在它在 data[0] 位置)
            self.data[0] = value
            
        # --- 情况 2: 写入位置在当前范围之后 ---
        elif internal_idx >= len(self.data):
            # 需要在 data 数组的 *后面* 补零
            num_append_zeros = internal_idx - len(self.data) + 1
            # 使用 np.pad 在后面补零
            self.data = np.pad(self.data, (0, num_append_zeros), 'constant')
            # 写入值 (现在它在 data[internal_idx] 或 data[-1] 位置)
            self.data[internal_idx] = value
            
        # --- 情况 3: 写入位置在当前范围内 ---
        else:
            self.data[internal_idx] = value

    def __str__(self):
        """用于 print() 函数，方便我们查看序列"""
        return f"SignalSequence(data={self.data}, n_start={self.n_start}, n_end={self.n_end})"
        
    def __repr__(self):
        """用于在控制台直接输出对象时显示"""
        return self. __str__()
    
    def copy(self):
        """
        返回此序列的一个副本。
        这对于防止在操作中修改原始序列非常重要。
        """
        # np.copy() 会创建一个数据的新副本
        return SignalSequence(np.copy(self.data), self.n0)
    
    def pad(self, pad_left, pad_right):
        """
        实现显式的前后补零 (要求 3a)
        :param pad_left: 在 n_start 前补 'pad_left' 个 0
        :param pad_right: 在 n_end 后补 'pad_right' 个 0
        :return: 一个新的 SignalSequence 实例
        """
        if pad_left < 0 or pad_right < 0:
            raise ValueError("补零长度必须是非负数")
        
        # 1. 使用 numpy.pad 高效补零
        new_data = np.pad(self.data, (pad_left, pad_right), 'constant', constant_values=0)
        
        # 2. 计算新的起始索引
        # 因为我们在数据数组的前面加了 pad_left 个元素，
        # 所以新的起始索引 n0 必须向左移动 pad_left 位。
        new_n0 = self.n0 - pad_left
        
        # 返回一个 *新* 的序列对象，不修改原始序列
        return SignalSequence(new_data, new_n0)

    def shift(self, k):
        """
        实现序列的移位 (要求 3b)
        y[n] = x[n - k]
        :param k: 移位的量
                  k > 0: 延迟 (shift right, 右移)
                  k < 0: 提前 (shift left, 左移)
        :return: 一个新的 SignalSequence 实例
        """
        # 移位操作不改变 data 数组的内容，只改变起始索引 n0
        # y[n] = x[n-k]
        # 举例: x[0] 的值，现在是 y[k] 的值
        # 即 y[n0 + k] = x[n0]
        # 所以 new_n0 = old_n0 + k
        
        # 返回一个 *新* 的序列对象，使用相同数据的副本
        return SignalSequence(self.data.copy(), self.n0 + k)
        
    def reverse(self):
        """
        实现序列的反转 (要求 3c)
        y[n] = x[-n]
        :return: 一个新的 SignalSequence 实例
        """
        # 1. 数据反转
        new_data = np.flip(self.data)
        
        # 2. 索引反转
        # 原始索引范围: [n_start, n_end]
        # 新索引范围: [-n_end, -n_start]
        # 例如: x[2] 变为 y[-2]
        # 新序列的第一个元素 (new_data[0]) 对应的是 x 的最后一个元素 (x[n_end])
        # y[new_n0] = x[n_end]
        # 根据 y[n] = x[-n]，我们有 y[-n_end] = x[n_end]
        # 所以 new_n0 = -n_end
        new_n0 = -self.n_end
        
        return SignalSequence(new_data, new_n0)
    
    def upsample(self, k):
        """
        实现序列拉伸 (上采样) (要求 3d)
        y[n] = x[n/k],  如果 n/k 是整数
               0,         其他
        :param k: (int) 上采样因子 (必须为正整数)
        :return: 一个新的 SignalSequence 实例
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("上采样因子 k 必须是正整数")
        
        # 1. 计算新序列的长度和起始/结束索引
        # 例: x = [1, 2], n0=0. (n=0, 1)
        #    k=2. y = [1, 0, 2], n0=0. (n=0, 1, 2)
        #    新长度 = (旧长度 - 1) * k + 1
        new_n0 = self.n0 * k
        new_n_end = self.n_end * k
        new_len = (self.n_end - self.n0) * k + 1
        
        # 2. 创建一个全零的新数据数组
        new_data = np.zeros(new_len, dtype=self.data.dtype)
        
        # 3. 计算原始数据在新数组中的位置
        # 原始数据 [0, 1, 2, ...] 对应新数组的 [0, k, 2k, ...]
        original_indices_in_new_data = np.arange(len(self.data)) * k
        
        # 4. 填充数据
        new_data[original_indices_in_new_data] = self.data
        
        return SignalSequence(new_data, new_n0)

    def downsample(self, k):
        """
        实现序列压缩 (下采样) (要求 3d)
        y[n] = x[n*k]
        :param k: (int) 下采样因子 (必须为正整数)
        :return: 一个新的 SignalSequence 实例
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("下采样因子 k 必须是正整数")

        # 1. 找到所有 n 是 k 的倍数的原始索引
        # 例: x, n_start=-2, n_end=2. indices = [-2, -1, 0, 1, 2]
        #    k=2. 采样的 n = [-2, 0, 2]
        all_n = self.indices
        sampled_n = all_n[all_n % k == 0]
        
        if len(sampled_n) == 0:
            # 如果没有点被采到，返回一个空序列
            return SignalSequence([], 0)

        # 2. 获取这些索引对应的数据
        # 使用 self[n] (我们的 __getitem__) 来安全地获取数据
        new_data = [self[n] for n in sampled_n]
        
        # 3. 计算新的起始索引
        # y[n] = x[n*k]
        # y[n_new] = x[n_old]  => n_new = n_old / k
        new_n0 = sampled_n[0] // k
        
        return SignalSequence(new_data, new_n0)

    def diff(self):
        """
        实现序列差分 (要求 3e)
        y[n] = x[n] - x[n-1]
        :return: 一个新的 SignalSequence 实例
        """
        # 1. 计算差分
        # y[n0] = x[n0] - x[n0 - 1]
        # 因为 x[n0 - 1] (越界) 被 __getitem__ 定义为 0
        # 所以我们需要在差分前，在 data 数组前补一个 0
        # np.diff([a, b, c]) = [b-a, c-b]
        # np.diff([a, b, c], prepend=0) = [a-0, b-a, c-b]
        new_data = np.diff(self.data, prepend=0)
        
        # 2. 起始索引不变
        return SignalSequence(new_data, self.n0)

    def accum(self):
        """
        实现序列累加 (要求 3e)
        y[n] = sum(x[k] for k = -inf to n)
        :return: 一个新的 SignalSequence 实例
        """
        # 1. 计算累加和 (Cumulative Sum)
        # y[n0] = x[n0]
        # y[n0+1] = x[n0] + x[n0+1]
        new_data = np.cumsum(self.data)
        
        # 2. 起始索引不变
        return SignalSequence(new_data, self.n0)

    def __add__(self, other):
        """
        实现序列加法 (要求 4a)
        y[n] = self[n] + other[n]
        :param other: (SignalSequence) 另一个信号序列
        :return: 一个新的 SignalSequence 实例
        """
        if not isinstance(other, SignalSequence):
            raise TypeError("操作数必须是 SignalSequence 类型")

        # 1. 确定新序列的索引范围 (并集)
        # 结果序列 y 的范围必须覆盖 x1 和 x2 的所有非零区域
        new_n_start = min(self.n_start, other.n_start)
        new_n_end = max(self.n_end, other.n_end)
        
        # 2. 创建一个新的、全为 0 的结果序列 y
        # (我们不能简单地复制，因为新序列的长度和 n0 都可能不同)
        new_len = new_n_end - new_n_start + 1
        new_data = np.zeros(new_len, dtype=np.promote_types(self.data.dtype, other.data.dtype))
        y = SignalSequence(new_data, new_n_start)
        
        # 3. 逐点相加
        # 遍历新序列 y 的每一个索引 n
        for n in y.indices:
            # y[n] = self[n] + other[n]
            # 这里的 self[n] 和 other[n] 会调用我们定义的 __getitem__
            # 自动处理越界返回 0 的情况，完美实现加法定义
            y[n] = self[n] + other[n]
            
        return y

    def __mul__(self, other):
        """
        实现序列乘法 (要求 4b)
        y[n] = self[n] * other[n]
        :param other: (SignalSequence) 另一个信号序列
        :return: 一个新的 SignalSequence 实例
        """
        if not isinstance(other, SignalSequence):
            raise TypeError("操作数必须是 SignalSequence 类型")

        # 1. 确定新序列的索引范围 (并集)
        # 注意：对于乘法，理论上只需要计算 *交集* 范围
        # 但为了与加法保持一致性，并处理 x[n]*0=0 的情况，使用并集更简单
        new_n_start = min(self.n_start, other.n_start)
        new_n_end = max(self.n_end, other.n_end)
        
        new_len = new_n_end - new_n_start + 1
        new_data = np.zeros(new_len, dtype=np.promote_types(self.data.dtype, other.data.dtype))
        y = SignalSequence(new_data, new_n_start)
        
        # 2. 逐点相乘
        for n in y.indices:
            # y[n] = self[n] * other[n]
            # 同样，__getitem__ 会自动处理越界 (返回 0)
            # x[n] * 0 = 0，完美实现乘法定义
            y[n] = self[n] * other[n]
            
        return y

    def convolve(self, h):
        """
        实现线性卷积 (要求 4c-i)
        y[n] = sum(x[k] * h[n-k] for k = -inf to +inf)
        
        不允许调用库函数
        
        :param h: (SignalSequence) 另一个信号序列
        :return: 一个新的 SignalSequence 实例
        """
        if not isinstance(h, SignalSequence):
            raise TypeError("操作数必须是 SignalSequence 类型")

        # 1. 确定 x 序列 (self) 和 h 序列 (h)
        x = self
        
        # 2. 确定 y 序列的索引范围
        y_n_start = x.n_start + h.n_start
        y_n_end = x.n_end + h.n_end
        
        # 3. 创建一个新的、全为 0 的结果序列 y
        y_len = y_n_end - y_n_start + 1
        new_data_type = np.promote_types(x.data.dtype, h.data.dtype)
        y = SignalSequence(np.zeros(y_len, dtype=new_data_type), y_n_start)
        
        # 4. 执行卷积求和
        # 外层循环: 遍历 y[n] 的所有索引 n
        for n in y.indices:
            # 内层循环: 遍历 x[k] 的所有索引 k
            # (k 的范围只需要在 x 的非零区域即可)
            
            # y[n] = ... + x[k-1]h[n-(k-1)] + x[k]h[n-k] + x[k+1]h[n-(k+1)] + ...
            sum_n = 0.0
            
            for k in x.indices:
                # 核心计算: x[k] * h[n - k]
                # x[k] 通过 x[k] (self.__getitem__) 获取
                # h[n - k] 通过 h[n - k] (other.__getitem__) 获取
                # __getitem__ 自动处理了所有越界情况 (返回 0)
                term = x[k] * h[n - k]
                sum_n += term
            
            # 将计算好的 y[n] 写入结果序列
            y[n] = sum_n
            
        return y
    
    def circular_convolve(self, h, N=None):
        """
        实现圆周卷积 (要求 4c-ii)
        y[n] = sum(x_p[k] * h_p[(n-k) % N] for k = 0 to N-1)
        
        注意: 圆周卷积的标准定义在 n = [0, N-1] 上。
        此方法将 *忽略* 两个序列的 n0，只使用它们的数据数组。
        
        :param h: (SignalSequence) 另一个信号序列
        :param N: (int, optional) 卷积周期。
                  如果 N=None, N = max(len(x), len(h)).
                  如果 N < max(len(x), len(h)), 会引发错误。
        :return: 一个新的 SignalSequence 实例 (n0 将为 0)
        """
        if not isinstance(h, SignalSequence):
            raise TypeError("操作数必须是 SignalSequence 类型")
            
        # 打印警告，说明 n0 被忽略
        if self.n0 != 0 or h.n0 != 0:
            print(f"[警告] circular_convolve: "
                  f"n0={self.n0} 和 n0={h.n0} 将被忽略。"
                  "圆周卷积假定 n 从 0 开始。")

        x_data = self.data
        h_data = h.data
        L = len(x_data)
        M = len(h_data)

        # 1. 确定周期 N
        if N is None:
            N = max(L, M)
        elif N < L or N < M:
            raise ValueError(f"指定的周期 N={N} "
                             f"小于序列长度 L={L} 或 M={M}，会导致时域混叠。")
        
        # 2. 补零到长度 N
        # (如果 N > L 或 N > M, np.pad 会自动补零)
        # (如果 N = L 或 N = M, 补零操作不执行)
        x_pad = np.pad(x_data, (0, N - L), 'constant')
        h_pad = np.pad(h_data, (0, N - M), 'constant')
        
        # 3. 创建结果数组
        y_data = np.zeros(N, dtype=np.promote_types(x_data.dtype, h_data.dtype))

        # 4. 执行圆周卷积求和
        # y[n] = sum(x_pad[k] * h_pad[(n - k) % N] for k in 0..N-1)
        for n in range(N):
            sum_n = 0.0
            for k in range(N):
                # 计算 h 的索引，带模 N 运算
                h_index = (n - k) % N
                sum_n += x_pad[k] * h_pad[h_index]
            y_data[n] = sum_n
            
        # 结果序列的 n0 始终为 0
        return SignalSequence(np.real(y_data), n0=0)
    
    def energy(self):
        """
        (辅助方法) 计算序列的总能量
        E = sum(x[n]^2)
        """
        # (self.data**2) 计算每个元素的平方
        # np.sum() 将它们全部相加
        return np.sum(self.data**2)

    def sliding_window_similarity(self, template):
        """
        实现滑动窗口相似性比对 (要求 4d-i)
        这等同于计算互相关 (Cross-Correlation):
        y[n] = sum(x[k] * template[k-n])
        
        这可以通过以下卷积实现:
        y[n] = x[n] * template[-n]
        
        :param template: (SignalSequence) 要匹配的模板序列 h
        :return: 一个新的 SignalSequence 实例，其值代表在每个位移 n 处的相似度
        """
        # 1. 互相关 = x 卷积 h 的反转
        # R_xh[n] = x[n] * h[-n]
        template_reversed = template.reverse()
        
        # 2. 我们已有的 convolve 方法完美地处理了所有索引
        return self.convolve(template_reversed)

    def normalized_similarity(self, template, epsilon=1e-10):
        """
        实现归一化的相似性比对 (要求 4d-ii)
        这等同于计算归一化互相关 (Normalized Cross-Correlation)
        
        y[n] = R_xh[n] / sqrt(E_h * E_x_window[n])
        
        :param template: (SignalSequence) 要匹配的模板序列 h
        :param epsilon: (float) 避免除以零的小值
        :return: 一个新的 SignalSequence 实例，其值在 [-1, 1] 之间
        """
        # 1. 计算分子: 互相关 R_xh[n]
        # (这与 4d-i 完全相同)
        y_corr = self.sliding_window_similarity(template)
        
        # 2. 计算分母中的 E_h (模板 h 的总能量)
        # 这是一个标量 (单个数字)
        h_energy = template.energy()
        
        # 3. 计算分母中的 E_x_window[n] (x 在滑动窗口内的能量)
        # 这是一个序列，与 y_corr[n] 长度相同
        
        # 3a. 创建一个与模板 h 形状相同的 "能量窗口" w
        w_data = np.ones_like(template.data)
        w = SignalSequence(w_data, template.n0)
        
        # 3b. 将 w 反转
        w_reversed = w.reverse()
        
        # 3c. 计算 x 的平方
        x_squared = self * self # 使用我们已有的 __mul__
        
        # 3d. 卷积 x^2 和 w_rev
        # 这会有效地在 x^2 上滑动一个全 1 的窗口，得到滑动能量
        x_window_energy_seq = x_squared.convolve(w_reversed)
        
        # 4. 计算最终的归一化序列
        
        # 此时 y_corr 和 x_window_energy_seq 具有完全相同的索引
        
        # E_h 是一个标量, x_window_energy_seq.data 是一个数组
        denominator = np.sqrt(x_window_energy_seq.data * h_energy) + epsilon
        
        y_data = y_corr.data / denominator
        
        # 结果序列的 n0 与互相关序列的 n0 相同
        return SignalSequence(y_data, y_corr.n0)





# ====================================================================
#
#   第 2 部分: 无限长序列 (随来随处理)
#   使用 Python 生成器 (Generators) 实现
#
# ====================================================================

import time

def infinite_source(stop_command="stop"):
    """
    (要求 2d) 模拟一个无限输入源
    它会不断请求用户输入，直到收到停止指令
    """
    print("\n--- 启动无限输入流 (输入 'stop' 结束) ---")
    while True:
        try:
            val_str = input("请输入下一个信号值: ")
            if val_str.lower() == stop_command:
                print("--- 停止无限输入流 ---")
                break # 停止生成
            
            yield float(val_str) # 产出当前值
        except ValueError:
            print("输入无效，请输入一个数字或 'stop'")
        except EOFError:
            break

def gen_delay(input_gen, k):
    """
    (要求 3b) 实现无限序列的延迟 (移位 k > 0)
    y[n] = x[n-k]
    """
    if k < 0:
        raise ValueError("延迟 k 必须为非负数 (k>=0)")
    
    # 1. 先准备 k 个 0 作为初始填充
    buffer = [0.0] * k
    
    # 2. 从输入流中读取
    for val in input_gen:
        # 产出 buffer 中的第一个值
        yield buffer.pop(0)
        # 将新值放入 buffer
        buffer.append(val)
    
    # 3. 输入流结束后，清空 buffer (对于有限输入流)
    while buffer:
        yield buffer.pop(0)

def gen_diff(input_gen):
    """
    (要求 3e) 实现无限序列的差分
    y[n] = x[n] - x[n-1]
    """
    # 状态：需要记住前一个值
    prev_val = 0.0
    
    for current_val in input_gen:
        yield current_val - prev_val
        prev_val = current_val # 更新状态

def gen_accum(input_gen):
    """
    (要求 3e) 实现无限序列的累加
    y[n] = y[n-1] + x[n]
    """
    # 状态：需要记住总和
    total_sum = 0.0
    
    for val in input_gen:
        total_sum += val
        yield total_sum

def gen_add(gen1, gen2):
    """
    (要求 4a) 实现两个无限序列的加法
    """
    # zip() 会自动从两个生成器中各取一个值，
    # 直到 *任何* 一个生成器耗尽
    for val1, val2 in zip(gen1, gen2):
        yield val1 + val2

def gen_convolve_finite(input_gen, h_finite_seq):
    """
    (要求 4c-i) 实现无限输入流 x[n] 与 *有限* 脉冲响应 h[n] 的卷积
    y[n] = sum(h[k] * x[n-k])
    
    :param input_gen: 无限输入流 (x[n])
    :param h_finite_seq: (SignalSequence) 有限的 h[n] (FIR 滤波器)
    """
    
    # 1. 检查 h 是否从 n=0 开始 (简化流式处理的假设)
    if h_finite_seq.n0 != 0:
        print(f"[警告] gen_convolve_finite: h[n] 的 n0={h_finite_seq.n0} "
              f"将被忽略。仅支持 h[n] 从 n=0 开始。")
    
    h_data = h_finite_seq.data
    h_len = len(h_data)
    
    # 2. 维护一个固定长度的缓冲区，模拟“滑动窗口”
    # 仅存储过去的 h_len 个输入值，确保操作是因果的
    x_buffer = [0.0] * h_len
    
    # 3. 开始流式处理
    for x_n in input_gen:
        # 更新状态：移除最旧的，加入最新的
        x_buffer.pop(0)
        x_buffer.append(x_n)
        
        # 乘累加运算 (MAC): 计算当前时刻的卷积值
        # y[n] = h[0]*x[n] + h[1]*x[n-1] + ... + h[M-1]*x[n-M+1]
        y_n = 0.0
        
        for k in range(h_len):
            # h[k] 对应 x_buffer 中的倒数第 k+1 个元素
            # x_buffer[h_len - 1 - k] 就是 x[n-k]
            y_n += h_data[k] * x_buffer[h_len - 1 - k]
        
        # 【关键修复】：yield 必须在 for x_n 循环的 *内部*
        # 每处理一个输入，就产出一个输出
        yield y_n

def gen_mul(gen1, gen2):
        """
        (要求 4b) 实现两个无限序列的乘法
        """
        # zip() 会自动从两个生成器中各取一个值，
        # 直到 *任何* 一个生成器耗尽
        for val1, val2 in zip(gen1, gen2):
            yield val1 * val2

def gen_upsample(input_gen, k):
    """
    (要求 3d) 实现无限序列的拉伸 (上采样)
    y[n] = x[n/k]
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("上采样因子 k 必须是正整数")
    
    for val in input_gen:
        # 1. 产出原始值
        yield val
        # 2. 产出 k-1 个 0
        for _ in range(k - 1):
            yield 0.0

def gen_downsample(input_gen, k):
    """
    (要求 3d) 实现无限序列的压缩 (下采样)
    y[n] = x[nk]
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("下采样因子 k 必须是正整数")
    
    while True:
        try:
            # 1. 产出当前值
            yield next(input_gen)
            # 2. 丢弃 (消耗) k-1 个值
            for _ in range(k - 1):
                next(input_gen) 
        except StopIteration:
            # 当 input_gen 耗尽时，退出
            break

def gen_sliding_window_similarity_causal(input_gen, h_finite_seq):
    """
    (要求 4d-i) "随来随处理" 的滑动窗口相似性 (因果版本)
    
    "即时操作" 无法实现非因果的互相关。
    我们实现一个因果的 "匹配滤波器" (Matched Filter):
    y[n] = x[n] * h_match[n]
    其中 h_match[n] = h[-n] (反转) 并平移到 n=0 开始。
    """
    # 1. 检查 h 是否从 n=0 开始 (因果滤波器的要求)
    if h_finite_seq.n0 != 0:
        print(f"[警告] gen_sliding_window_similarity: h[n] 的 n0={h_finite_seq.n0} "
                f"将被忽略。模板假定从 n=0 开始。")
    
    # 2. h_match[n] = h[-n] (反转) 并平移到 n=0 开始
    h_match_data = np.flip(h_finite_seq.data)
    h_match = SignalSequence(h_match_data, n0=0)
    
    # 3. 调用卷积生成器 (它会正确处理 h_match 的 n0=0)
    # (我们必须将它转换回生成器，否则它只是一个函数调用)
    for val in gen_convolve_finite(input_gen, h_match):
        yield val

def gen_normalized_similarity(input_gen, template, epsilon=1e-10):
    """
    (要求 4d-ii) "随来随处理" 的归一化相似性 (NCC)
    
    这需要两个并行的处理管线，因此实现非常复杂。
    y[n] = R_xh[n] / sqrt(E_h * E_x_window[n])
    """
    
    # --- 0. 检查模板 ---
    if template.n0 != 0:
        print(f"[警告] gen_normalized_similarity: 模板 h[n] 的 n0={template.n0} "
                f"将被忽略。模板假定从 n=0 开始。")
    
    # 1. 拆分输入流，一个用于分子，一个用于分母
    # (itertools.tee 会缓存数据，确保两个管线都能收到)
    gen_num, gen_den = itertools.tee(input_gen)
    
    # --- 2. 分子管线 (Numerator Pipeline) ---
    # y_num[n] = R_xh[n] (因果版本)
    num_pipeline = gen_sliding_window_similarity_causal(gen_num, template)
    
    # --- 3. 分母管线 (Denominator Pipeline) ---
    
    # 3a. 创建一个 (x[n])^2 的流
    x_squared_gen = (val**2 for val in gen_den)
    
    # 3b. 创建能量窗口 w (全 1, 长度与 template 相同, n0=0)
    w_data = np.ones_like(template.data)
    w = SignalSequence(w_data, n0=0) # n0=0
    
    # 3c. 计算滑动窗口能量: (x^2 * w)[n]
    # (注意: w 是对称的, w.reverse() 还是 w 都可以)
    sliding_energy_gen = gen_convolve_finite(x_squared_gen, w)
    
    # 3d. 模板 h 的总能量 (这是一个常数)
    h_energy = template.energy()
    
    # 3e. 组装分母流
    den_pipeline = (np.sqrt(val * h_energy) + epsilon for val in sliding_energy_gen)

    # --- 4. 组装最终管线 ---
    # 逐点相除
    for num_val, den_val in zip(num_pipeline, den_pipeline):
        yield num_val / den_val










# --- 用于测试的“主”代码块 ---
if __name__ == "__main__":
    
    print("--- 1. 测试初始化 (要求 2a, 2b) ---")
    # 新数据: 一个三角波 [1, 2, 3, 2, 1]，起始 n0=-2
    x_data = [1, 2, 3, 2, 1]
    x_orig = SignalSequence(x_data, n0=-2)
    print(f"原始序列 x_orig: {x_orig}")
    print(f"n_start (应为 -2): {x_orig.n_start}, n_end (应为 2): {x_orig.n_end}")

    print("\n--- 2. 测试任意位置读取 (要求 2c) ---")
    print(f"读取 x_orig[-2] (应为 1): {x_orig[-2]}")
    print(f"读取 x_orig[0] (应为 3): {x_orig[0]}")
    print(f"读取 x_orig[5] (越界, 应为 0): {x_orig[5]}")

    print("\n--- 3. 测试任意位置写入 (要求 2c) ---")
    x = x_orig.copy() # 复制一个副本进行修改
    print(f"写入前 x[0] (应为 3): {x[0]}")
    x[0] = 5
    print(f"写入后 x[0] (应为 5): {x[0]}")
    print(f"写入前 x[1] (应为 2): {x[1]}")
    x[1] = x[1] - 1
    print(f"写入后 x[1] (应为 1): {x[1]}")
    print(f"修改后的序列 x: {x}") # data=[1, 2, 5, 1, 1], n0=-2

    print("\n--- 4. 测试写入时自动扩展 (补零) ---")
    x_ext = x_orig.copy() # 拿一个干净的副本
    # a) 向后扩展
    x_ext[4] = 10
    print(f"向后扩展 (x_ext[4]=10): {x_ext}")
    print(f"x_ext[3] (自动补零): {x_ext[3]}")
    # b) 向前扩展
    x_ext[-4] = -5
    print(f"向前扩展 (x_ext[-4]=-5): {x_ext}")
    print(f"x_ext[-3] (自动补零): {x_ext[-3]}")
    # 最终 x_ext: data=[-5, 0, 1, 2, 3, 2, 1, 0, 10], n_start=-4, n_end=4

    print("\n--- 5. 测试显式补零 (要求 3a) ---")
    x_padded = x_ext.pad(pad_left=1, pad_right=2)
    print(f"原始序列 x_ext: {x_ext}")
    print(f"补零后 (前1, 后2): {x_padded}")
    print(f"新 n_start (应为 -4-1=-5): {x_padded.n_start}")
    print(f"新 n_end (应为 4+2=6): {x_padded.n_end}")

    print("\n--- 6. 测试序列移位 (要求 3b) ---")
    # k=2 (延迟, 右移)
    x_delayed = x_ext.shift(2)
    print(f"延迟 k=2 (右移): {x_delayed}")
    print(f"x_ext[-4] (值为-5) 移动到 x_delayed[-2]: {x_delayed[-2]}")
    # k=-1 (提前, 左移)
    x_advanced = x_ext.shift(-1)
    print(f"提前 k=-1 (左移): {x_advanced}")
    print(f"x_ext[-4] (值为-5) 移动到 x_advanced[-5]: {x_advanced[-5]}")

    print("\n--- 7. 测试序列反转 (要求 3c) ---")
    # x_ext: data=[-5...10], n_start=-4, n_end=4
    x_reversed = x_ext.reverse()
    print(f"反转后 y: {x_reversed}")
    print(f"新 n_start (应为 -4): {x_reversed.n_start}, 新 n_end (应为 4): {x_reversed.n_end}")
    print(f"y[0] (y[0]=x[0]=3): {x_reversed[0]}")
    print(f"y[4] (y[4]=x[-4]=-5): {x_reversed[4]}")
    print(f"y[-4] (y[-4]=x[4]=10): {x_reversed[-4]}")

    # --- 新的测试基准 ---
    x_test = SignalSequence([1, -1, 1], n0=0)
    print(f"\n--- 8. 测试拉伸 (上采样 k=2) (要求 3d) ---")
    # x: n=[0, 1, 2], data=[1, -1, 1]
    x_up = x_test.upsample(2)
    print(f"原始序列: {x_test}")
    print(f"上采样 k=2 后 (应为 [1, 0, -1, 0, 1], n0=0): {x_up}")

    print(f"\n--- 9. 测试压缩 (下采样 k=2) (要求 3d) ---")
    # x: n=[0, 1, 2], data=[1, -1, 1]
    # 采样的 n = [0, 2]
    # y: n=[0, 1], data=[x[0], x[2]] = [1, 1]
    x_down = x_test.downsample(2)
    print(f"原始序列: {x_test}")
    print(f"下采样 k=2 后 (应为 [1, 1], n0=0): {x_down}")

    print(f"\n--- 10. 测试差分 (要求 3e) ---")
    # x: n=[0, 1, 2], data=[1, -1, 1]
    # y[0]=x[0]-x[-1]=1-0=1
    # y[1]=x[1]-x[0]=-1-1=-2
    # y[2]=x[2]-x[1]=1-(-1)=2
    x_diff = x_test.diff()
    print(f"原始序列: {x_test}")
    print(f"差分后 (应为 [1, -2, 2], n0=0): {x_diff}")

    print(f"\n--- 11. 测试累加 (要求 3e) ---")
    # x: n=[0, 1, 2], data=[1, -1, 1]
    # y[0]=1
    # y[1]=1 + (-1)=0
    # y[2]=0 + 1=1
    x_acc = x_test.accum()
    print(f"原始序列: {x_test}")
    print(f"累加后 (应为 [1, 0, 1], n0=0): {x_acc}")

    print("\n--- 12. 测试序列加法 (要求 4a) ---")
    x1 = SignalSequence([1, 1, 1], n0=-1)
    x2 = SignalSequence([2, 2], n0=0)
    # n: [-1, 0, 1]
    # y[-1] = 1+0=1
    # y[0] = 1+2=3
    # y[1] = 1+2=3
    y_add = x1 + x2
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")
    print(f"y_add (x1 + x2) (应为 [1, 3, 3], n0=-1): {y_add}")

    print("\n--- 13. 测试序列乘法 (要求 4b) ---")
    # n: [-1, 0, 1]
    # y[-1] = 1*0=0
    # y[0] = 1*2=2
    # y[1] = 1*2=2
    y_mul = x1 * x2
    print(f"y_mul (x1 * x2) (应为 [0, 2, 2], n0=-1): {y_mul}")

    print("\n--- 14. 测试线性卷积 (要求 4c-i) ---")
    x_conv = SignalSequence([1, 2, 1], n0=0)
    h_conv = SignalSequence([1, -1], n0=0)
    # y: n=[0, 1, 2, 3]
    # y[0]=1*1=1
    # y[1]=1*(-1)+2*1=1
    # y[2]=2*(-1)+1*1=-1
    # y[3]=1*(-1)=-1
    y_conv = x_conv.convolve(h_conv)
    print(f"x = {x_conv}")
    print(f"h = {h_conv}")
    print(f"y_conv (x * h) (应为 [1, 1, -1, -1], n0=0): {y_conv}")

    print("\n--- 15. 测试圆周卷积 (要求 4c-ii) ---")
    x_circ = SignalSequence([1, 2, 3, 4], n0=0)
    h_circ = SignalSequence([1, 1], n0=1) # n0=1 将被忽略
    # N = max(4, 2) = 4
    # x_pad = [1, 2, 3, 4]
    # h_pad = [1, 1, 0, 0] (h的n0=1被忽略)
    print(f"x = {x_circ}")
    print(f"h = {h_circ}")
    # y[0] = 1*1 + 2*0 + 3*0 + 4*1 = 5
    # y[1] = 1*1 + 2*1 + 3*0 + 4*0 = 3
    # y[2] = 1*0 + 2*1 + 3*1 + 4*0 = 5
    # y[3] = 1*0 + 2*0 + 3*1 + 4*1 = 7
    y_circ = x_circ.circular_convolve(h_circ)
    print(f"y_circ (N=4) (应为 [5, 3, 5, 7], n0=0): {y_circ}")

    print("\n--- 16. 测试滑动窗口相似性 (互相关) (要求 4d-i) ---")
    x_sig = SignalSequence([0, 1, 1, 0, -1, -1, 0], n0=0)
    h_tpl = SignalSequence([1, 1, 0], n0=0)
    print(f"信号 x = {x_sig}")
    print(f"模板 h = {h_tpl}")
    y_sim = x_sig.sliding_window_similarity(h_tpl)
    # R[1] = x[1]h[0] + x[2]h[1] = 1*1 + 1*1 = 2
    # R[4] = x[4]h[0] + x[5]h[1] = -1*1 + -1*1 = -2
    print(f"y_sim (互相关) = {y_sim}")
    print(f"y_sim[1] (应为峰值 2.0): {y_sim[1]}")
    print(f"y_sim[4] (应为谷值 -2.0): {y_sim[4]}")

    print("\n--- 17. 测试归一化相似性 (NCC) (要求 4d-ii) ---")
    # E_h = 1^2 + 1^2 + 0^2 = 2
    # n=1: R[1]=2. x_win=[1, 1, 0]. E_x=2. y[1]=2/sqrt(2*2)=1.0
    # n=4: R[4]=-2. x_win=[-1, -1, 0]. E_x=2. y[4]=-2/sqrt(2*2)=-1.0
    y_ncc = x_sig.normalized_similarity(h_tpl)
    print(f"y_ncc (归一化) = {y_ncc}")
    print(f"y_ncc[1] (应为 1.0): {y_ncc[1]}")
    print(f"y_ncc[4] (应为 -1.0): {y_ncc[4]}")