import numpy as np
import struct


class Tensor:
    """
    这个类用于存储神经网络中的张量，包括数据、梯度、运算等信息。
    它支持基本的加法和乘法运算，并提供了激活函数和反向传播方法。

    其中主要包含两类方法：
    1. 前向传播（运算）方法：用于执行加法和乘法等运算。
    2. 反向传播方法：用于计算梯度。

    注意：这个类并没有提供改变神经网络参数的方法，因为它只存储数据和梯度。
          如果需要改变神经网络参数，应该在主程序中自行编写。

    """

    def __init__(self, data: np.array, requires_grad=False):
        # 用于将向量形式转化为Nx1矩阵形式
        if data.ndim == 1:
            self.data = data.reshape(-1, 1)
        else:
            self.data = data

        # 注意 梯度也需要统一形式
        self.grad = None  # 梯度值（初始为None）
        self.requires_grad = requires_grad  # 是否需要计算梯度
        self.op = None  # 生成该节点的运算（如加法、乘法）
        self.parents = []  # 输入节点列表（父节点，构成计算图的边）

    @staticmethod
    def activate_function(x):
        """
        激活函数：暂时使用RELU函数
        也可以使用其他激活函数，如ReLU、tanh等。
        :param x: 输入值
        :return: 激活后的值
        """
        return 1 / (1 + np.exp(-x))  # sigmoid函数实现
        # return np.where(x > 0, x, 0.01 * x)  # ReLU函数实现

    @staticmethod
    def d_activate_function(x):
        """
        激活函数的导数：sigmoid函数的导数
        也可以使用其他激活函数的导数。
        :param x: 输入值
        :return: 激活函数的导数
        """
        s = 1 / (1 + np.exp(-x))  # 复用sigmoid函数计算
        return s * (1 - s)  # 导数公式 σ'(x) = σ(x)(1-σ(x))
        # return np.where(x > 0, 1, 0.01)  # ReLU函数的导数实现

    def __add__(self, other):
        # 加法运算
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=True)
            out.op = 'add'
            out.parents = [self, other]
            return out
        else:  # 估计是用不到了
            out = Tensor(self.data + other, requires_grad=True)
            out.op = 'add'
            out.parents = [self]
            return out

    def add_forward(self, other):
        return self + other

    def add_backward(self):
        # 加法反向传播
        if self.op == "add":

            # parent 0
            if self.parents[0].requires_grad:
                if self.parents[0].grad is None:
                    self.parents[0].grad = self.grad
                else:
                    self.parents[0].grad += self.grad

            # parent 1
            if self.parents[1].requires_grad:
                if self.parents[1].grad is None:
                    self.parents[1].grad = self.grad
                else:
                    self.parents[1].grad += self.grad

        else:
            print("Error: add_backward only works for add operation.")

    def __sub__(self, other):
        # 减法运算
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, requires_grad=True)
            out.op = 'sub'
            out.parents = [self, other]
            return out
        else:
            out = Tensor(self.data - other, requires_grad=True)
            out.op = 'sub'
            out.parents = [self]
            return out

    def sub_forward(self, other):
        return self - other

    def sub_backward(self):

        if self.op == "sub":
            if self.parents[0].requires_grad:

                if self.parents[0].grad is None:
                    self.parents[0].grad = self.grad
                else:
                    self.parents[0].grad += self.grad
            if self.parents[1].requires_grad:
                if self.parents[1].grad is None:
                    self.parents[1].grad = -self.grad
                else:
                    self.parents[1].grad += -self.grad
        else:
            print("Error: sub_backward only works for sub operation.")

    def __mul__(self, other):
        """
        乘法运算

        这里的乘法运算，是对应元素相乘，而不是矩阵乘法。不要和dot()方法搞混了。

        矩阵乘法可以使用dot_forward()方法实现。
        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=True)
            out.op = 'mul'
            out.parents = [self, other]
            return out
        else:
            # 处理数乘的情况

            # 这里就是将数字转化为元素全部相同的，相同形状的张量进行乘法运算
            # temp储存了输入标量所对应的张量
            temp = Tensor(np.full(self.data.shape, other))
            out = Tensor(self.data * temp, requires_grad=True)
            out.op = 'mul'
            out.parents = [self, temp]
            return out

    def mul_forward(self, other):
        """
        乘法运算的前向传播
        这里的乘法运算，是对应元素相乘，而不是矩阵乘法。
        矩阵乘法可以使用dot_forward()方法实现。
        :param other:要乘的数
        :return:
        """
        return self * other

    def mul_backward(self):
        """
        乘法运算的反向传播
        这里的乘法运算，是对应元素相乘，而不是矩阵乘法。
        矩阵乘法可以使用dot_backward()方法实现。
        :return:
        """
        if self.op == "mul":

            if self.parents[0].requires_grad:
                if self.parents[0].grad is None:
                    self.parents[0].grad = self.grad * self.parents[1].data
                else:
                    self.parents[0].grad += self.grad * self.parents[1].data

            if self.parents[1].requires_grad:
                if self.parents[1].grad is None:
                    self.parents[1].grad = self.grad * self.parents[0].data
                else:
                    self.parents[1].grad += self.grad * self.parents[0].data
        else:
            print("Error: mul_backward only works for mul operation.")

    def __pow__(self, power, modulo=None):
        """
        幂运算 尽量输入整数
        注意：power只能是整数，否则反向传播无法对指数进行求导
        :param power: 指数
        :param modulo:
        :return:
        """
        # 幂运算
        if isinstance(power, Tensor):
            print("暂不支持Tensor的幂运算")
            out = Tensor(self.data ** power.data, requires_grad=True)
            out.op = 'pow'
            out.parents = [self, power]
            return out
        else:
            out = Tensor(self.data ** power, requires_grad=True)
            out.op = 'pow'
            out.parents = [self, power]
            return out

    def pow_forward(self, other):
        """
        幂运算的前向传播

        暂时只支持与标量相乘的幂运算，即self为Tensor，other为float or int
        :param other: 指数
        :return:
        """
        return self ** other

    def pow_backward(self):
        """
        幂运算的反向传播
        暂时只支持与标量相乘的幂运算，即self为Tensor，other为float or int
        :return:
        """
        if self.op == "pow":
            if self.parents[0].requires_grad:
                if self.parents[0].grad is None:
                    self.parents[0].grad = self.grad * self.parents[1] * (
                            self.parents[0].data ** (self.parents[1] - 1))
                else:
                    self.parents[0].grad += self.grad * self.parents[1] * (
                            self.parents[0].data ** (self.parents[1] - 1))

    def activate_forward(self):
        """
        激活函数的前向传播
        :return:
        """
        out = Tensor(self.activate_function(self.data), requires_grad=True)
        out.op = 'activate'
        out.parents = [self]
        return out

    def activate_backward(self):
        """
        激活函数的反向传播
        :return:
        """
        if self.op == "activate":
            if self.parents[0].requires_grad:
                if self.parents[0].grad is None:
                    self.parents[0].grad = self.grad * self.d_activate_function(self.parents[0].data)
                else:
                    self.parents[0].grad += self.grad * self.d_activate_function(self.parents[0].data)
        else:
            print("Error: activate_backward only works for activate operation.")

    def dot_forward(self, other):
        """
        矩阵点乘
        以及矩阵乘向量
        向量点乘向量

        注意：这里指的是矩阵运算，而不是逐个元素相乘。不要和mul()方法搞混了。

        逐个元素相乘可以使用mul_forward()方法实现。

        self在左是矩阵，other在右是向量
        :param other: 需要乘的向量
        :return: 返回一个向量
        """
        # 需要考虑向量相乘
        if self.data.shape[1] == 1 and other.data.shape[1] == 1:
            out = Tensor(np.dot(self.data.T, other.data), requires_grad=True)
            out.op = 'dot'
            out.parents = [self, other]
            return out

        else:
            # 矩阵在左
            out = Tensor(np.dot(self.data, other.data), requires_grad=True)
            out.op = 'dot'
            out.parents = [self, other]
            return out

    def dot_backward(self):
        """
        矩阵乘向量的反向传播
        :return:
        """
        if self.op == "dot":
            # 处理父节点0（权重矩阵）的梯度
            if self.parents[0].requires_grad:
                grad_parent0 = np.dot(self.grad, self.parents[1].data.T)

                # 下面这个只是个打上去的补丁，如果self.grad为一个数字，
                # 那么上面那条语句就会导致梯度矩阵形状错误
                if self.grad.shape == (1, 1):
                    grad_parent0 = grad_parent0.reshape(-1, 1)

                if self.parents[0].grad is None:
                    self.parents[0].grad = grad_parent0
                else:
                    self.parents[0].grad += grad_parent0

            # 处理父节点1（输入向量）的梯度
            if self.parents[1].requires_grad:
                grad_parent1 = np.dot(self.parents[0].data.T, self.grad)

                if self.parents[1].grad is None:
                    self.parents[1].grad = grad_parent1
                else:
                    self.parents[1].grad += grad_parent1
        else:
            print("Error: dot_backward only works for dot operation.")

    def convolution_forward(self, kernel, stride=1):
        """
        卷积运算，专门用于卷积神经网络
        :param kernel: 卷积核
        :param stride: 步长
        :return: 返回卷积后的结果
        """
        if isinstance(kernel, Tensor):
            # 计算输出形状
            out_height = (self.data.shape[0] - kernel.data.shape[0]) // stride + 1
            out_width = (self.data.shape[1] - kernel.data.shape[1]) // stride + 1

            # 初始化输出矩阵
            data_temp = np.zeros((out_height, out_width))

            # 卷积运算
            # 根据原数据 卷积核 步长决定卷积结果
            for i in range(0, self.data.shape[0] - kernel.data.shape[0] + 1, stride):
                for j in range(0, self.data.shape[1] - kernel.data.shape[1] + 1, stride):
                    # 卷积运算
                    temp = self.data[i:i + kernel.data.shape[0], j:j + kernel.data.shape[1]] * kernel.data
                    # 求和
                    temp = np.sum(temp)
                    # 存储到卷积后的结果中
                    data_temp[i][j] = temp

            out = Tensor(data_temp, requires_grad=True)
            out.op = 'convolution'
            out.parents = [self, kernel, stride]

            return out

        else:
            print("error: kernel is not a Tensor")

    def convolution_backward(self):
        """
        卷积运算的反向传播
        :return:
        """
        if self.op == "convolution":
            if self.parents[0].requires_grad:
                # 对输入的图像矩阵进行处理
                # 这里的处理方式是将卷积核旋转180度，然后进行卷积运算
                # 首先需要将自己的梯度扩展出去

                # temp_self_grad是这次卷积计的输入部分
                temp_self_grad = np.pad(self.grad, pad_width=self.parents[1].data.shape[0] - 1,
                                        mode='constant', constant_values=0)

                # 然后将卷积核旋转180度
                temp_kernel = np.rot90(self.parents[1].data, 2)

                # 然后进行卷积运算
                # 这是卷积计算的输出值
                # 长度：输入值减去卷积核长度+1
                data_temp = np.zeros(((temp_self_grad.shape[0] - temp_kernel.shape[0] + 1) // self.parents[2],
                                     (temp_self_grad.shape[1] - temp_kernel.shape[1] + 1) // self.parents[2]))

                for i in range(0, temp_self_grad.shape[0] - temp_kernel.shape[0] + 1, self.parents[2]):
                    for j in range(0, temp_self_grad.shape[1] - temp_kernel.shape[1] + 1, self.parents[2]):
                        # 卷积运算
                        temp = temp_self_grad[i:i + temp_kernel.shape[0], j:j + temp_kernel.shape[1]] * temp_kernel
                        # 求和
                        temp = np.sum(temp)
                        # 存储到卷积后的结果中
                        data_temp[i][j] = temp

                # 将卷积后的结果存储到parents[0]的梯度数值中
                if self.parents[0].grad is None:
                    self.parents[0].grad = data_temp
                else:
                    self.parents[0].grad += data_temp

            # 接下来处理卷积核的梯度
            if self.parents[1].requires_grad:
                # 卷积核梯度初始化
                if self.parents[1].grad is None:
                    self.parents[1].grad = np.zeros_like(self.parents[1].data)

                # 可能需要遍历卷积核

                for i in range(self.parents[1].data.shape[0]):
                    for j in range(self.parents[1].data.shape[1]):
                        # 缓存卷积结果的长度
                        res_len = self.data.shape[0]

                        # 卷积核对结果的影响还是蛮大的，卷积核的梯度为卷积结果与原图相应的部分相乘后求和
                        grad_temp = self.parents[0].data[i:i + res_len, j:j + res_len] * self.grad
                        self.parents[1].grad[i][j] = np.sum(grad_temp)

                # 暂时不支持多个梯度返回到一个卷积核
                """
                if self.parents[1].grad is None:
                    self.parents[1].grad = self.grad * self.parents[0].data
                else:
                    self.parents[1].grad += self.grad * self.parents[0].data
                """

        else:

            print("Error: convolution_backward only works for convolution operation.")

    def max_pooling_forward(self, pooling_size=(2, 2)):
        """
        max池化运算，专门用于卷积神经网络
        :param pooling_size: 池化核大小
        """
        # 池化输出结果缓存
        pooling_temp = np.zeros((self.data.shape[0] // pooling_size[0], self.data.shape[1] // pooling_size[1]))

        # 记录每个窗口最大值位置的掩码矩阵，用于反向传播
        max_mask = np.zeros_like(self.data)

        for i in range(0, self.data.shape[0], pooling_size[0]):
            for j in range(0, self.data.shape[1], pooling_size[1]):
                # 获取当前池化窗口
                window = self.data[i:i + pooling_size[0], j:j + pooling_size[1]]

                # 找到最大值和其在窗口内的相对位置
                max_val = np.max(window)
                max_pos = np.unravel_index(np.argmax(window), window.shape)

                # 存储池化结果
                pooling_temp[i // pooling_size[0], j // pooling_size[1]] = max_val

                # 在原始数据位置记录最大值位置(1表示最大值位置)
                max_mask[i + max_pos[0], j + max_pos[1]] = 1

        # 创建池化后的Tensor对象
        out = Tensor(pooling_temp, requires_grad=True)
        out.op = 'max_pooling'
        out.parents = [self, max_mask, pooling_size]
        return out

    def max_pooling_backward(self):
        """
        max池化运算的反向传播
        :return:
        """
        if self.op == "max_pooling":
            if self.parents[0].requires_grad:
                # 初始化梯度矩阵
                grad_input = np.zeros_like(self.parents[0].data)

                # 获取池化参数
                pooling_size = self.parents[2]
                max_mask = self.parents[1]

                # 将梯度分配到前向传播时最大值的位置
                for i in range(0, self.parents[0].data.shape[0], pooling_size[0]):
                    for j in range(0, self.parents[0].data.shape[1], pooling_size[1]):
                        # 获取当前池化区域
                        region = max_mask[i:i + pooling_size[0], j:j + pooling_size[1]]
                        # 将梯度分配到最大值位置
                        grad_input[i:i + pooling_size[0], j:j + pooling_size[1]] = \
                            (region * self.grad[i // pooling_size[0], j // pooling_size[1]])

                # 更新父节点的梯度
                if self.parents[0].grad is None:
                    self.parents[0].grad = grad_input
                else:
                    self.parents[0].grad += grad_input

    def auto_backward(self):
        """
        通过self.op标签中的字符串决定反向传播类型

        注意：自动反向传播只支持add, sub, mul, pow, activate, dot, convolution, max_pooling操作

        ps:不过嘛 auto的东西还是尽量不要用了啦

        :return:
        """
        if self.op == "add":
            self.add_backward()
        elif self.op == "sub":
            self.sub_backward()
        elif self.op == "mul":
            self.mul_backward()
        elif self.op == "pow":
            self.pow_backward()
        elif self.op == "activate":
            self.activate_backward()
        elif self.op == "dot":
            self.dot_backward()
        elif self.op == "convolution":
            self.convolution_backward()
        elif self.op == "max_pooling":
            self.max_pooling_backward()
        else:
            print("Error: auto_backward only works for add, sub, mul, pow, activate, dot operation.")


class FCNN:
    # Fully Connected Neural Network
    def __init__(self, depth: int, layer_size: tuple, input_size: int):
        # 注意：layer_size最后一层应当为10
        # depth是屎山，应该去掉的，懒得改了

        # 初始化网络
        self.label = None
        self.input = None  # 输入向量，784维（28x28图像展开）
        self.cost = 0  # 损失值

        # 定义隐藏层层数
        self.depth = depth  # 网络深度（层数）
        self.layer_size = layer_size  # 每层的神经元数量

        # layers作为二维数组 存储隐藏层 内部元素为np数组
        self.layers = []  # 存储每一层的激活值

        # 初始化每一层的激活值为Tensor
        for _ in range(depth):
            self.layers.append(Tensor(np.zeros(layer_size[_]), requires_grad=False))

        # 初始化权重矩阵
        self.weights = []  # 权重矩阵列表

        for _ in range(depth):
            if _ == 0:
                self.weights.append(Tensor(np.random.randn(layer_size[_], input_size) * np.sqrt(2. / input_size), requires_grad=True))
            else:
                self.weights.append(
                    Tensor(np.random.randn(layer_size[_], layer_size[_ - 1]) * np.sqrt(2. / layer_size[_ - 1]),
                           requires_grad=True))

        # 初始化偏置向量
        self.biases = []  # 偏置向量列表

        for _ in range(depth):
            self.biases.append(Tensor(np.zeros(layer_size[_]).reshape(-1, 1), requires_grad=True))

    def forward(self, input, input_required_grad: bool = False):
        """
        前向传播函数
        :param input_required_grad: 选择是否计算输入的梯度，默认为False，用于兼容其他类型网络
        :param input: 作为神经网络的输入向量，可以是numpy数组或Tensor对象 注意：Tensor对象一定要reshape一下哟
        :return: 无
        """
        # 处理输入数据
        if isinstance(input, Tensor):
            self.input = input  # 保存输入数据
        else:
            self.input = Tensor(input.reshape(-1, 1), requires_grad=input_required_grad)  # 保存输入数据

        # 处理第一层神经
        """self.layers[0] = ((self.weights[0].dot_forward(Tensor(input, requires_grad=input_required_grad))
                           + self.biases[0])
                          .activate_forward())"""
        self.layers[0] = ((self.weights[0].dot_forward(self.input)
                           + self.biases[0])
                          .activate_forward())
        self.layers[0].requires_grad = True

        # 处理后续层
        for _ in range(1, self.depth):
            self.layers[_] = ((self.weights[_].dot_forward(self.layers[_ - 1])
                               + self.biases[_])
                              .activate_forward())
            self.layers[_].requires_grad = True

    def backward(self, label: np.ndarray, learning_rate=0.1):
        """
        反向传播函数
        :param label: 以one-hot编码的形式给出正确的标签
        :param learning_rate: 学习率
        :return: 无
        """
        self.label = Tensor(label.reshape(-1, 1), requires_grad=False)

        # 计算损失值对输出层的梯度
        cost_temp = []  # 长度为2

        # temp 1 存储最后一层与目标值之差
        # temp 2 存储 temp 1 的平方
        # cost 存储 temp 2 中数据之和
        cost_temp.append(self.layers[-1] - self.label)
        cost_temp.append(cost_temp[0].pow_forward(2))
        self.cost = cost_temp[1].dot_forward(Tensor(np.ones(10), requires_grad=False))
        self.cost.grad = np.array([1]).reshape(-1, 1)

        # 反向传播
        self.cost.dot_backward()
        cost_temp[1].pow_backward()
        cost_temp[0].sub_backward()

        # 反向传播至各隐藏层（从倒数第二层开始）
        for i in range(self.depth - 1, -1, -1):
            self.layers[i].activate_backward()  # 返回到激活前 激活前的对象获取梯度
            self.layers[i].parents[0].add_backward()  # 返回到Wa 和 b
            self.layers[i].parents[0].parents[0].dot_backward()  # Wa 返回到 W 和 a

        # 更新权重和偏置
        for i in range(self.depth):
            self.weights[i].data -= learning_rate * self.weights[i].grad
            self.biases[i].data -= learning_rate * self.biases[i].grad

    def erase_grad(self):
        """
        清空梯度
        :return:
        """
        # 清空梯度
        for i in range(self.depth):
            self.weights[i].grad = None
            self.biases[i].grad = None
            self.layers[i].grad = None
        if not isinstance(self.cost, int):
            self.cost.grad = None
        # 清空输入数据
        self.input.grad = None


def read_images(filepath):
    """
    读取MNIST图像文件
    :param filepath: 文件路径
    :return: 图像数据 (样本数, 行, 列)
    """
    # 读取MNIST图像文件
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, "Invalid image file format"
        # 一次性读取所有图像数据并转换为numpy数组
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        return images.reshape(num, rows, cols)  # 转换为三维数组 (样本数, 行, 列)


def read_labels(filepath):
    """
    读取MNIST标签文件
    :param filepath: 文件路径
    :return: 标签数据 (样本数,)
    """
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 0x00000801, "Invalid label file format"
        # 读取所有标签数据并转换为numpy数组
        return np.frombuffer(f.read(num), dtype=np.uint8)


if __name__ == '__main__':
    # DEBUG = False  # 没啥用，用于显示过程

    # 读取MNIST数据
    train_images = read_images('data\\train-images.idx3-ubyte')
    train_labels = read_labels('data\\train-labels.idx1-ubyte')

    test_images = read_images('data\\t10k-images.idx3-ubyte')
    test_labels = read_labels('data\\t10k-labels.idx1-ubyte')

    network = FCNN(depth=2, layer_size=(10,10), input_size=784)

    # 训练60000张图片
    for i in range(6000):
        one_hot = np.zeros(10)
        one_hot[train_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(train_images[i][j])

        network.forward(Tensor(np.array(temp) / 255, requires_grad=True))

        # network.forward(np.array(temp) / 255, True)
        network.backward(np.array(one_hot), 0.003)
        network.erase_grad()
        print(network.cost)

    # 10000张图片用于验证
    # count 用于记录正确的数量
    count = 0
    for i in range(1000):
        one_hot = np.zeros(10)
        one_hot[test_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(test_images[i][j])

        network.forward(np.array(temp) / 255)

        if np.argmax(network.layers[-1].data) == test_labels[i]:
            count += 1

    # 输出正确率
    print(count / 1000)
