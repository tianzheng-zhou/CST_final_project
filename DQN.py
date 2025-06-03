import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



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
        激活函数：暂时使用Relu函数
        也可以使用其他激活函数，如sigmoid、tanh等。
        :param x: 输入值
        :return: 激活后的值
        """
        return np.maximum(0, x)  # ReLU函数

    @staticmethod
    def d_activate_function(x):
        """
        激活函数的导数：Relu函数的导数
        也可以使用其他激活函数的导数。
        :param x: 输入值
        :return: 激活函数的导数
        """
        """ReLU的导数"""
        return np.where(x > 0, 1, 0)

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

    def auto_backward(self):
        """
        通过self.op标签中的字符串决定反向传播类型

        注意：自动反向传播只支持add, sub, mul, pow, activate, dot操作

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
        else:
            print("Error: auto_backward only works for add, sub, mul, pow, activate, dot operation.")


class FCNN:
    def __init__(self, input_size, depth: int, layer_size: tuple):
        self.depth = depth
        self.layer_size = layer_size
        self.layers = []
        self.weights = []
        self.biases = []

        for _ in range(depth):
            if _ == 0:
                self.layers.append(Tensor(np.zeros(layer_size[_]), requires_grad=False))
                self.weights.append(Tensor(np.random.randn(layer_size[_], input_size) * np.sqrt(2. / input_size), requires_grad=True))
            else:
                self.layers.append(Tensor(np.zeros(layer_size[_]), requires_grad=False))
                self.weights.append(Tensor(np.random.randn(layer_size[_], layer_size[_-1]) * np.sqrt(2. / layer_size[_-1]), requires_grad=True))
            self.biases.append(Tensor(np.zeros((layer_size[_], 1)), requires_grad=True))

    def forward(self, input):
        if not isinstance(input, Tensor):
        # 确保输入是列向量 (n_features, 1)
            if input.ndim == 1:
                input = input.reshape(-1, 1)
            elif input.ndim == 2 and input.shape[0] == 1:
                input = input.T  # 转置行向量为列向量
            self.input = Tensor(input, requires_grad=True)
        else:
            self.input = input

        self.layers[0] = (self.weights[0].dot_forward(self.input) + self.biases[0]).activate_forward()
        for i in range(1, self.depth):
            if i == self.depth - 1:
                self.layers[i] = self.weights[i].dot_forward(self.layers[i-1]) + self.biases[i]
            else:
                self.layers[i] = (self.weights[i].dot_forward(self.layers[i-1]) + self.biases[i]).activate_forward()
        return self.layers[-1]

    def clear_gradients(self):
        for layer in self.layers + self.weights + self.biases:
            if hasattr(layer, 'grad'):
                layer.grad = None



# 迷宫生成函数（沿用之前逻辑）
def generate_maze(width, height):
    # 初始化迷宫（0表示通路，-1表示墙壁，初始时全是墙壁）
    maze = [[-1 for _ in range(width)] for _ in range(height)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 用DFS方法生成迷宫
    def dfs(x, y):
        maze[y][x] = 0  # 标记为通路
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = x + 2*dx, y + 2*dy
            # 检查新的状态点是否在边界内
            if 0 <= new_x < width and 0 <= new_y < height:
                if maze[new_y][new_x] == -1:  # 如果是未访问的墙壁
                    maze[y+dy][x+dx] = 0      # 打通当前墙
                    dfs(new_x, new_y)         # 递归访问

    # 随机一个起点，并从这个点开始生成迷宫
    start_x, start_y = random.randint(0,width-1),random.randint(0,height-1)
    dfs(start_x, start_y)
    
    # 设置目标点
    while True:
        target_x, target_y = random.randint(0, width-1), random.randint(0, height-1)
        if maze[target_y][target_x] == 0 and (target_x, target_y) != (start_x, start_y):  #确保目标点与起点不重合
            maze[target_y][target_x] = 1
            return maze, (target_x, target_y) ,(start_x,start_y)

# 智能体类（修改动作映射为0-3）
class Agent:
    def __init__(self, position):
        self.x, self.y = position
        self.actions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # 上、下、右、左（对应动作0-3）
    
    def move(self, action, maze_array):
        dx, dy = self.actions[action]
        new_x, new_y = self.x + dx, self.y + dy
        if 0<=new_x< maze_array.shape[1] and 0<=new_y<maze_array.shape[0] and maze_array[new_y][new_x] != -1:
            return new_x, new_y
        else:
            return self.x,self.y
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.next_idx = 0
    
    def add(self, experience):
        """添加经验到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience
            self.next_idx = (self.next_idx + 1) % self.capacity
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        """返回当前缓冲区的大小"""
        return len(self.buffer)
# DQN训练类（基于FCNN和Tensor）
class DQNTrainer:
    def __init__(self, state_size, action_size, hidden_layers=(64,64), learning_rate=0.001, gamma=0.95,  epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        # 主网络和目标网络
        self.policy_net = FCNN(
            input_size=state_size,
            depth=len(hidden_layers)+1,
            layer_size=(*hidden_layers, action_size)
        )
        self.target_net = FCNN(
            input_size=state_size,
            depth=len(hidden_layers)+1,
            layer_size=(*hidden_layers, action_size)
        )
        self._sync_target_network()

        self.memory = ReplayBuffer(10000)  # 经验回放缓冲区
    def choose_action(self, state, training=True):
            self.epsilon = max(self.epsilon_min, self.epsilon **self.epsilon_decay)  # 衰减epsilon
            if random.random() < self.epsilon and training:
                return random.randint(0, self.action_size-1)
            else:
                state_normalized = state / (np.array([1,1,1,1])*width)  # 关键：测试时也需要归一化
                state_reshaped = state_normalized.reshape(-1, 1)
                self.policy_net.forward(state_reshaped)
                q_values = self.policy_net.layers[-1].data.flatten()
                return np.argmax(q_values)
    def _sync_target_network(self):
        """同步主网络到目标网络"""
        for w_p, w_t in zip(self.policy_net.weights, self.target_net.weights):
            w_t.data = w_p.data.copy()
        for b_p, b_t in zip(self.policy_net.biases, self.target_net.biases):
            b_t.data = b_p.data.copy()

    def remember(self, state, action, reward, next_state, done):
        
        # 存储经验及其优先级
        self.memory.add((state, action, reward, next_state, done))

    def train_step(self, batch_size=32):
        if len(self.memory.buffer) < batch_size:
            return
        
        # 初始化梯度为零
        for param in self.policy_net.weights + self.policy_net.biases:
            if param.grad is not None:
                param.grad.fill(0)
        
        samples= self.memory.sample(batch_size)
        
        # 处理每个样本
        for state, action, reward, next_state, done in samples:
            # 确保状态是正确的形状 (4, 1)
            state_tensor = Tensor(state.reshape(-1, 1), requires_grad=False)
            next_state_tensor = Tensor(next_state.reshape(-1, 1), requires_grad=False)
            
            # 计算目标Q值
            self.target_net.forward(next_state_tensor)
            max_next_q = np.max(self.target_net.layers[-1].data)
            
            # 计算当前Q值
            self.policy_net.forward(state_tensor)
            current_q = self.policy_net.layers[-1].data.flatten()
            
            # 更新目标动作的Q值
            target_q = current_q.copy()
            target_q[action] = reward + self.gamma * max_next_q * (1 - done)
            target_q_tensor = Tensor(target_q.reshape(-1, 1), requires_grad=False)
            
            # 计算损失
            loss = self.policy_net.layers[-1] - target_q_tensor
            loss = loss.pow_forward(2)  # 对每个Q值计算平方误差
            
            # 计算标量损失 (求和)
            loss_sum = loss.dot_forward(Tensor(np.ones_like(loss.data), requires_grad=False))
            loss_sum.grad = np.ones_like(loss_sum.data)  # 标量损失的梯度为1
            
            # 反向传播 (累积梯度)
            loss_sum.auto_backward()
        
        # 统一更新参数 (除以batch_size实现平均梯度)
        for w in self.policy_net.weights:
            if w.grad is not None:
                # 计算梯度范数
                grad_norm = np.linalg.norm(w.grad)
                if grad_norm > 1.0:  # 设置阈值
                    w.grad = w.grad / grad_norm  # 归一化梯度
                w.data -= (self.learning_rate / batch_size) * w.grad
        for b in self.policy_net.biases:
            if b.grad is not None:
                b.data -= (self.learning_rate / batch_size) * b.grad
        
        # 清空梯度
        self.policy_net.clear_gradients()
# 训练流程
def train_dqn(maze, start_pos, target_pos, episodes=2000, batch_size=32):
    maze_array = np.array(maze)
    state_size = 4
    action_size = 4  # 4种动作
    trainer = DQNTrainer(state_size, action_size, hidden_layers=(64,64))
    agent = Agent(start_pos)
    rewards_history = []
    

    for episode in range(episodes):
        agent.x, agent.y = start_pos
        state = np.array([agent.x,agent.y,target_pos[0],target_pos[1]])
        total_reward = 0
        done = False
        while not done and len(trainer.memory.buffer) < trainer.memory.capacity:
            action = trainer.choose_action(state)
            reward = 0
            
            x, y = agent.move(action, maze_array)
            next_state = np.array([x,y,target_pos[0],target_pos[1]])

            if maze_array[y, x] == 1:  # 到达目标
                reward+= 100
                done = True
            if maze_array[y, x] == -1:
                reward-=10
            else:
                reward-=1
            trainer.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            agent.x ,agent.y = x, y
                # 定期训练和同步目标网络
        if len(trainer.memory.buffer) >= batch_size:
            trainer.train_step(batch_size)
            if episode % 10 == 0:
                trainer._sync_target_network()

        rewards_history.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Reward: {total_reward}, Memory: {len(trainer.memory.buffer)}")

    return trainer, rewards_history

# 测试流程
def test_dqn(trainer, maze, start_pos, target_pos,):
    trainer._sync_target_network()  # 确保网络同步
    trainer.epsilon = 0.1
    maze_array = np.array(maze)
    agent = Agent(start_pos)
    state = np.array([agent.x,agent.y,target_pos[0],target_pos[1]])
    path = [(agent.x, agent.y)]
    done = False

    while not done and len(path) < 100:
        action = trainer.choose_action(state, training=False)
        x, y = agent.move(action, maze_array)
        while (x,y) == (agent.x,agent.y):
           action = trainer.choose_action(state, training=True)
           x,y = agent.move(action,maze_array)

        path.append((x, y))
        maze_array[agent.y,agent.x] = -1
        state = np.array([x,y,target_pos[0],target_pos[1]])
        q_values = trainer.policy_net.forward(state).data.flatten()
        action = np.argmax(q_values)

        if maze_array[y, x] == 1:
            done = True
            print("找到目标！路径长度：", len(path))
        elif maze_array[y, x] == -1:
            print("撞墙，路径失败")
            break
        agent.x,agent.y = x,y
    return path
# 绘制最终路径
def plot_final_path(maze, path, start_pos, target_pos):
    # 创建迷宫数组
    maze_array = np.array(maze)
    
    # 标记起点和终点
    maze_array[start_pos[1]][start_pos[0]] = 2  # 起点
    maze_array[target_pos[1]][target_pos[0]] = 1  # 终点
    
    # 创建颜色映射
    cmap = ListedColormap(['black', 'white', 'green', 'red'])
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    plt.imshow(maze_array, cmap=cmap, vmin=-1, vmax=3)
    
    # 添加网格
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, maze_array.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, maze_array.shape[0], 1), [])
    
    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'bo-', linewidth=2, markersize=8)
        plt.plot(path_x[0], path_y[0], 'yo', markersize=12, label='Start')  # 起点
        plt.plot(path_x[-1], path_y[-1], 'ro', markersize=12, label='End')  # 终点
    
    # 添加图例
    plt.legend(loc='upper right')
    
    # 添加标题
    plt.title(f'Maze Path (Length: {len(path)})')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('maze_final_path.png')
    plt.show()
    print("最终路径图已保存为 'maze_final_path.png'")

# 主函数
if __name__ == "__main__":
    width, height = 9,9
    state_size = 4
    maze, target_pos, start_pos = generate_maze(width, height)
    maze_array = np.array(maze)
    maze_array[start_pos[1]][start_pos[0]] = 2  # 标记起点
    print("生成的迷宫：")
    print(maze_array)
    # 训练DQN
    print("\n开始训练...")
    trainer, rewards = train_dqn(maze, start_pos, target_pos, episodes=2000)
    # 测试
    print("\n开始测试...")
    path = test_dqn(trainer, maze, start_pos, target_pos)
    print("路径：", path)
    plot_final_path(maze, path, start_pos, target_pos)