import numpy as np
import matplotlib.pyplot as plt

# 1. Define Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def entropy_loss(s):
    # Add eps to prevent log(0)
    eps = 1e-8
    return -(s * np.log(s + eps) + (1 - s) * np.log(1 - s + eps))

def entropy_gradient_wrt_x(x):
    """
    Calculate the gradient of Loss with respect to the raw input x: dL/dx
    Derivation: dL/dx = (s - (1-s)) * s * (1-s) = (2s - 1) * s * (1-s) 
    In terms of x: dL/dx = -x * s * (1-s) is a simplified trend visualization.
    Mathematically for entropy minimization, it pushes x away from 0.
    """
    s = sigmoid(x)
    # The gradient dL/dx for entropy minimization pushes s towards 0 or 1
    return (2 * s - 1) * s * (1 - s)

# 2. Prepare Data
x = np.linspace(-6, 6, 500) # Simulating MLP output (Logit)
s = sigmoid(x)              # Simulating Picker score (Score)
loss = entropy_loss(s)      # Calculate Entropy Loss
grad = entropy_gradient_wrt_x(x) # Calculate Gradient

# 3. Start Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Loss vs Score 
# Meaning: Higher uncertainty (closer to 0.5) results in higher loss
plt.subplot(1, 3, 1)
plt.plot(s, loss, color='blue', lw=2)
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.title("Entropy Loss vs Score\n(Peak at 0.5: Maximum Uncertainty)", fontsize=10)
plt.xlabel("Score (s)")
plt.ylabel("Loss Value")
plt.grid(True, alpha=0.3)

# Plot 2: Gradient vs Logit 
# Meaning: Gradient acts as a force. 
# In Gradient Descent: x = x - learning_rate * grad
# Positive grad pushes x left (towards 0), Negative grad pushes x right (towards 1)
plt.subplot(1, 3, 2)
plt.plot(x, grad, color='green', lw=2)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='red', linestyle='--', alpha=0.5)
plt.title("Gradient (dL/dx) vs Logit (x)\n(Force: Positive pushes Left, Negative pushes Right)", fontsize=10)
plt.xlabel("Logit (x)")
plt.ylabel("Gradient Value")
plt.fill_between(x, 0, grad, where=(grad>0), color='red', alpha=0.1, label='Push to 0')
plt.fill_between(x, 0, grad, where=(grad<0), color='blue', alpha=0.1, label='Push to 1')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Update Vector Field
# Simulating the direction of parameter updates
plt.subplot(1, 3, 3)
update_direction = -grad # Direction of gradient descent
plt.quiver(x[::25], np.zeros_like(x[::25]), update_direction[::25], np.zeros_like(x[::25]), 
           angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.6)
plt.axvline(0, color='red', linestyle='--', alpha=0.5)
plt.title("Update Direction (x = x - eta * grad)\n(Trend: Escaping the 0.5 Region)", fontsize=10)
plt.xlabel("Logit (x)")
plt.yticks([]) # Hide Y axis
plt.xlim(-6, 6)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz_entropy_loss.png')
plt.show()