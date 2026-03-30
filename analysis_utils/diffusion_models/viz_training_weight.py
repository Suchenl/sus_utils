'''
    加权代码, 来自DiffSynth-Studio

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing
'''


import torch
import matplotlib.pyplot as plt

def get_weights_from_your_code():
    # 模拟 self.timesteps (假设训练时通常是 0 到 1000)
    steps = 1000
    timesteps = torch.linspace(0, steps, steps)
    
    # --- 以下完全按照你给的代码逻辑 ---
    x = timesteps
    # 1. 核心高斯公式
    y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
    # y = torch.exp(-2 * ((x - steps / 4) / steps) ** 2)
    
    # 2. 减去最小值，让曲线底部对齐 0
    y_shifted = y - y.min()
    
    # 3. 归一化：保证所有权重的总和等于 steps (1000)
    bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
    
    # 4. 如果长度不等于 1000 的修正 (此处假设 len 等于 1000，则跳过)
    if len(timesteps) != 1000:
        bsmntw_weighing = bsmntw_weighing * (len(timesteps) / steps)
        bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
    
    return timesteps, bsmntw_weighing

# 获取数据
x_vals, y_vals = get_weights_from_your_code()

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(x_vals.numpy(), y_vals.numpy(), label='Training Weight (bsmntw_weighing)', color='blue')
plt.fill_between(x_vals.numpy(), y_vals.numpy(), alpha=0.1, color='blue')

plt.title("Training Weight Curve")
plt.xlabel("Timestep (x)")
plt.ylabel("Weight (y)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('training_weight_curve-div4.png')  # 保存图像到文件
# plt.show()
