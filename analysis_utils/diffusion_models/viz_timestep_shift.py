import numpy as np
import matplotlib.pyplot as plt

def flux_shift(sigmas, shift):
    ''' Also Wan shift '''
    return shift * sigmas / (1 + (shift - 1) * sigmas)

# 模拟原始线性 sigmas (从 1 到 0)
x = np.linspace(0, 1, 100)

plt.figure(figsize=(8, 6))

# 画出不同 shift 值的对比
shifts = [0.1, 0.5, 1.0, 3.0, 5.0, 10]
for s in shifts:
    y = flux_shift(x, s)
    label = f'shift={s}' + (' (Default)' if s==3.0 else '')
    plt.plot(x, y, label=label)

plt.title("Flux/Wan Sigma Time Shift Transformation")
plt.xlabel("Original Linear Sigma (0.0 to 1.0)")
plt.ylabel("Shifted Sigma")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("flux_sigma_shift.png")


'''
    加噪声逻辑: sigma 越大(timestep越大), 加的noise越多
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
'''