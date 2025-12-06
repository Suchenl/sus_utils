import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model (nn.Module): The model to analyze.
        trainable_only (bool): If True, only count trainable parameters.
    Returns:
        int: Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def estimate_activation_memory_by_hook(model: nn.Module, input_size: tuple, dtype=torch.float32) -> float:
    """
    [DEPRECATED METHOD] Estimate memory usage by capturing module outputs via hooks.
    
    NOTE: This method significantly UNDERESTIMATES memory usage because it CANNOT see
    temporary tensors created inside forward methods (e.g., attention matrices).
    It is kept here for educational purposes to contrast with the profiler method.
    
    Args:
        model (nn.Module): The model to analyze.
        input_size (tuple): Input tensor size (B, C, H, W).
        dtype: Torch dtype (default: float32).
    Returns:
        float: Estimated memory usage of module outputs in MB.
    """
    # Get data type size in bytes
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    
    # Store activation sizes
    activation_sizes = []

    def hook_fn(_, inp, out):
        if isinstance(out, torch.Tensor):
            activation_sizes.append(out.numel() * dtype_size)
        elif isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.numel() * dtype_size)

    hooks = []
    for layer in model.modules():
        # Avoid hooking container modules
        if not isinstance(layer, nn.ModuleList) and not isinstance(layer, nn.Sequential) and len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a dummy forward pass
    device = next(model.parameters()).device
    dummy_input = torch.zeros(*input_size, dtype=dtype, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    total_activations = sum(activation_sizes)
    return total_activations / (1024 ** 2)  # Convert to MB

def profile_peak_memory(model: nn.Module, input_size: tuple, dtype=torch.float32) -> float:
    """
    [SUPERIOR METHOD] Accurately measure the peak GPU memory usage during a forward pass.
    
    This method directly queries the CUDA driver for the maximum memory allocated,
    providing a true measure of the model's memory footprint during inference.

    Args:
        model (nn.Module): The model to analyze.
        input_size (tuple): Input tensor size (B, C, H, W).
        dtype: Torch dtype (default: float32).
    Returns:
        float: Peak allocated memory in MB.
    """
    device = next(model.parameters()).device
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Peak memory profiling is only for GPU.")
        return 0.0

    # Prepare dummy input and model
    model.eval()
    dummy_input = torch.zeros(*input_size, dtype=dtype, device=device)
    
    # Reset CUDA memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Run forward pass without gradients
    with torch.no_grad():
        _ = model(dummy_input)
        
    # Get the peak memory allocated from the CUDA driver
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    torch.cuda.empty_cache()
    
    return peak_memory_bytes / (1024 ** 2) # Convert to MB


def model_summary(model: nn.Module, input_size: tuple, dtype=torch.float32):
    """
    Print a comprehensive summary of model parameters and memory usage.
    
    This function provides a comparison between the old hook-based estimation
    and the new, more accurate peak memory profiling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Parameter Count ---
    params_m = count_parameters(model) / 1e6

    # --- Memory Estimation (Old Method) ---
    hook_mem_mb = estimate_activation_memory_by_hook(model, input_size, dtype)
    
    # --- Memory Profiling (New, Accurate Method) ---
    peak_mem_mb = profile_peak_memory(model, input_size, dtype)
    
    param_mem_mb = (params_m * 1e6 * torch.tensor([], dtype=dtype).element_size()) / (1024**2)

    print("="*50)
    print("Model Performance Summary")
    print("="*50)
    print(f"Input Shape: {list(input_size)}")
    print(f"Data Type:   {dtype}")
    print(f"Device:      {device}")
    print("-" * 50)
    print(f"Parameters:              {params_m:.2f} M")
    print(f"Parameter Memory:        {param_mem_mb:.2f} MB")
    print("-" * 50)
    print("Forward Pass Memory Analysis (Inference Mode):")
    print(f"  [Hook-based] Act. Est.:  {hook_mem_mb:.2f} MB (Underestimates, ignores temps)")
    print(f"  [Profiler] Peak VRAM:    {peak_mem_mb:.2f} MB (Accurate, includes all tensors)")
    print("="*50)
    print("Note: 'Peak VRAM' is the most important metric for determining if the model will fit on your GPU.")


# =================================================================================
# EXAMPLE USAGE
# =================================================================================
# This block should be runnable if you have the ResAttnUNet class available.
if __name__ == "__main__":
    # Mock ResAttnUNet for demonstration purposes if the actual class is not available
    from core.modules.unet_res_attn import ResAttnUNet
    # --- Configuration ---
    # Try changing window_size and observe the change in 'Peak VRAM'
    # For example, compare ws=8 vs ws=16 vs ws=32. You will now see a difference!

    model = ResAttnUNet(
        in_channels=3,
        out_channels=6,
        base_channels=32,
        channel_mult=(1, 2, 4),
        n_res_blocks=2,
        attn_in_stages=(0, 1, 2),
        attn_heads=4,
        window_size=16 
    )

    # --- Run the Analysis ---
    model_summary(model, input_size=(2, 3, 540, 960))

    # --- Example of how to see the effect of window_size ---
    print("\n--- Testing Effect of Window Size ---")
    
    for ws in [4, 8, 16, 32]:
        print(f"\nAnalyzing with window_size = {ws}...")
        model_ws = ResAttnUNet(
            in_channels=3, out_channels=6, base_channels=32,
            channel_mult=(1, 2, 4), n_res_blocks=2,
            attn_in_stages=(1, 2), attn_heads=4, window_size=ws
        ).cuda()
        peak_mem = profile_peak_memory(model_ws, input_size=(2, 3, 540, 960))
        print(f"-> Peak Memory: {peak_mem:.2f} MB")