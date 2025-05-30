from AutomaticDifferentiation import Tensor, read_labels, read_images, FCNN
import numpy as np


class CNN:
    def __init__(self):
        # 卷积核大小，池化大小

        self.kernel_size = (5, 3)
        self.pooling_size = (2, 2)

        # 输入的图片
        self.input = None

        # 卷积层
        self.conv_layers = None

        # 全连接层，调用FCNN类
        self.fc = FCNN(1, (10,), 25)
        self.fcnn_input = None

        # 保存卷积层到全连接层的形状，更好地连接卷积层以及全连接层
        self.shape = None

        self.kernel = []

        for i in self.kernel_size:
            # 初始化卷积核
            # 这里使用np中的随机数生成，大小为i*i，范围为low ~ high
            self.kernel.append(Tensor(np.random.uniform(low=-1, high=1, size=(i, i)), True))

        self.convolutional_layer = []

    def forward(self, x: np.array):
        """
        前向传播
        :param x: 输入的图片，大小为28*28，类型为np.array
        :return:
        """
        # 初始化输入
        self.input = Tensor(x, False)

        # 卷积和池化操作
        # 当然，顺序和次数也可以魔改

        # 下面的代码看上去有点屎山，不要介意哈
        self.conv_layers = []
        # 卷积操作
        self.conv_layers.append(self.input.convolution_forward(self.kernel[0], 1))
        # 激活
        self.conv_layers.append(self.conv_layers[0].activate_forward())
        # 池化操作
        self.conv_layers.append(self.conv_layers[1].max_pooling_forward(self.pooling_size))

        # 同上
        self.conv_layers.append(self.conv_layers[2].convolution_forward(self.kernel[1], 1))
        self.conv_layers.append(self.conv_layers[3].activate_forward())
        self.conv_layers.append(self.conv_layers[4].max_pooling_forward(self.pooling_size))

        # 连接全连接层
        self.shape = self.conv_layers[-1].data.shape

        # fcnn_input是对卷积层最后一层的引用
        self.fcnn_input = self.conv_layers[-1]
        # 更改形状以适配全连接层 注意：反向传播时要改回形状，同时也要改回梯度的形状
        self.fcnn_input.data = self.fcnn_input.data.reshape(-1, 1)

        # 全连接层forward
        self.fc.forward(self.fcnn_input)

    def backward(self, y: np.array, learning_rate: float = 0.001):
        """
        反向传播
        :param learning_rate: 学习率，默认为0.001
        :param y: 标签，大小为10，类型为np.array，one-hot编码
        :return:
        """
        # 全连接层反向
        self.fc.backward(y, learning_rate/5)

        # 更改形状以适配卷积层
        self.fcnn_input.data = self.fcnn_input.data.reshape(self.shape)
        self.fcnn_input.grad = self.fcnn_input.grad.reshape(self.shape)

        # 卷积层反向
        self.conv_layers[5].max_pooling_backward()
        self.conv_layers[4].activate_backward()
        self.conv_layers[3].convolution_backward()
        self.conv_layers[2].max_pooling_backward()
        self.conv_layers[1].activate_backward()
        self.conv_layers[0].convolution_backward()

        # 卷积核更新
        for i in range(len(self.kernel)):
            self.kernel[i].data -= learning_rate * self.kernel[i].grad

    def erase_grad(self):
        """
        清空梯度
        :return:
        """
        self.fc.erase_grad()
        # 清空梯度
        for i in range(len(self.kernel)):
            self.kernel[i].grad = None
        self.input.grad = None
        for i in self.conv_layers:
            i.grad = None


if __name__ == '__main__':
    # 读取数据
    train_images = read_images('data\\train-images.idx3-ubyte')
    train_labels = read_labels('data\\train-labels.idx1-ubyte')

    test_images = read_images('data\\t10k-images.idx3-ubyte')
    test_labels = read_labels('data\\t10k-labels.idx1-ubyte')
    # 初始化网络
    net = CNN()
    # 训练
    count = 0
    for i in range(6000):
        one_hot = np.zeros(10)
        one_hot[train_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(train_images[i][j])

        net.forward((np.array(temp) / 255).reshape(28, 28))
        net.backward(np.array(one_hot), 1.5*(count/6000))
        net.erase_grad()
        count += 1
        print(count)

    count = 0
    for i in range(1000):
        one_hot = np.zeros(10)
        one_hot[test_labels[i]] = 1
        temp = []
        for j in range(28):
            temp += list(test_images[i][j])

        net.forward((np.array(temp) / 255).reshape(28, 28))
        if np.argmax(net.fc.layers[-1].data) == test_labels[i]:
            count += 1

    print(count / 1000)
