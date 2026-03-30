import numpy as np
import matplotlib.pyplot as plt

# 1. Core Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_entropy(s):
    eps = 1e-8
    return -(s * np.log(s + eps) + (1 - s) * np.log(1 - s + eps))

def get_total_loss(x, lam):
    s = sigmoid(x)
    loss_ent = get_entropy(s)
    loss_l1 = lam * s
    return loss_ent + loss_l1

def get_total_gradient(x, lam):
    """
    dL/dx = (dL/ds) * (ds/dx)
    dL_ent/ds = -x, dL_l1/ds = lam
    ds/dx = s * (1-s)
    Final: dL/dx = (lam - x) * s * (1-s)
    """
    s = sigmoid(x)
    return (lam - x) * s * (1 - s)

# 2. Data Preparation
x = np.linspace(-6, 12, 1000) # Extended range to see the shift
s = sigmoid(x)
lambdas = [0, 0.1, 0.2, 0.5, 1, 3, 5, 7] # Different sparsity weights to visualize
colors = plt.cm.plasma(np.linspace(0, 0.8, len(lambdas)))

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Total Loss Curves
for lam, color in zip(lambdas, colors):
    total_loss = get_total_loss(x, lam)
    ax1.plot(x, total_loss, label=f'lambda={lam}', color=color, lw=2)
    
    # Mark the equilibrium (peak) for each lambda
    # Equilibrium is at x = lambda
    peak_val = get_total_loss(lam, lam)
    ax1.scatter(lam, peak_val, color=color, s=40, zorder=5)
    ax1.axvline(lam, color=color, linestyle=':', alpha=0.3)

ax1.set_title("Total Loss vs Logit (x)\n[Higher lambda shifts the peak to the right]", fontsize=12)
ax1.set_xlabel("Logit (x)")
ax1.set_ylabel("Loss Value (Entropy + lambda * s)")
ax1.legend()
ax1.grid(True, alpha=0.2)

# Subplot 2: Gradient Curves (The "Pushing Force")
for lam, color in zip(lambdas, colors):
    grad = get_total_gradient(x, lam)
    ax2.plot(x, grad, label=f'lambda={lam}', color=color, lw=2)
    
    # Highlight the zero-crossing (Decision Boundary)
    ax2.axhline(0, color='black', lw=1, alpha=0.5)
    ax2.scatter(lam, 0, color=color, s=40, zorder=5)

# Decoration for Gradient Plot
ax2.set_title("Total Gradient (dL/dx) vs Logit (x)\n[Zero-crossing = Survival Threshold]", fontsize=12)
ax2.set_xlabel("Logit (x)")
ax2.set_ylabel("Gradient Value")
ax2.fill_between(x, -0.25, 0.25, where=(x < 0), color='red', alpha=0.02)
ax2.text(-4, 0.15, "Global Kill Zone", color='red', alpha=0.5, fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.2)
ax2.set_ylim(-0.3, 0.3) # Limit for better visibility of the "force"

plt.tight_layout()
plt.savefig('viz_entropy_l1_loss.png', dpi=300)
