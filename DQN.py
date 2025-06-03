import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from AutomaticDifferentiation import Tensor, FCNN

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
        self.policy_net.erase_grad()
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