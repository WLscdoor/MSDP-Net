import torch
import torch.nn.functional as F
import math

def add_gaussian_noise(
    x: torch.Tensor,
    snr_range: tuple = (10, 30),  # SNR range (in dB)
) -> torch.Tensor:
    """
    Add Gaussian white noise with random SNR to a 1D signal.
    Args:
        x: Input signal of shape [1, L] or [L].
        snr_range: Signal-to-noise ratio range in dB, e.g., (10, 30) means from 10 dB to 30 dB.
    Returns:
        noisy_x: Noisy signal with the same shape as input.
    """
    # Ensure input shape is [1, L]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2, "Input should be a 1D signal of shape [1, L] or [L]."
    
    # Randomly sample SNR (in dB) and convert to linear scale
    snr_db = torch.empty(1).uniform_(*snr_range).item()
    snr_linear = 10 ** (snr_db / 10)
    
    # Compute signal power
    signal_power = torch.mean(x ** 2)
    
    # Compute noise power based on SNR
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian white noise
    noise = torch.randn_like(x) * math.sqrt(noise_power)
    
    # Add noise to signal
    noisy_x = x + noise
    
    return noisy_x

def random_circular_shift(x: torch.Tensor, max_shift: int = 32):
    """
    Apply random circular shift to a 1D signal (elements shifted beyond bounds wrap around).
    Args:
        x: Input tensor of shape [1, 256].
        max_shift: Maximum absolute shift amount.
    Returns:
        shifted_x: Shifted signal of shape [1, 256].
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=1)

def random_zoom(
    x: torch.Tensor, 
    zoom_range: tuple = (0.8, 1.2),
    mode: str = 'linear'
) -> torch.Tensor:
    """
    Apply random center-based zoom (scaling) to a 1D signal of arbitrary length.
    
    Args:
        x: Input signal of shape [1, L] or [L].
        zoom_range: Scaling factor range (e.g., (0.8, 1.2) means scale down to 80% or up to 120%).
        mode: Interpolation mode ('linear', 'nearest', 'cubic').
        
    Returns:
        zoomed_x: Scaled signal with the same shape as input.
    """
    # Ensure input shape is [1, L]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2, "Input should be a 1D signal of shape [1, L] or [L]."
    
    L = x.size(1)
    zoom_factor = torch.empty(1).uniform_(*zoom_range).item()
    if zoom_factor == 1.0:
        return x.clone()
    
    # Compute new length after scaling
    new_len = int(L * zoom_factor)
    zoomed_x = F.interpolate(x.unsqueeze(1), size=new_len, mode=mode).squeeze(1)
    
    if new_len > L:  # Zoom in: crop center part
        start = (new_len - L) // 2
        zoomed_x = zoomed_x[:, start:start+L]
        if zoomed_x.size(1) < L:  # Pad if needed to avoid boundary issues
            zoomed_x = F.pad(zoomed_x, (0, L - zoomed_x.size(1)))
    else:  # Zoom out: symmetrically pad
        pad_left = (L - new_len + 1) // 2
        pad_right = L - new_len - pad_left
        zoomed_x = F.pad(zoomed_x, (pad_left, pad_right))
    
    return zoomed_x  # Maintain input shape [1, L]

def random_shift(x: torch.Tensor, max_shift: int = 32):
    """
    Apply random non-circular shift (zero-padded) to a 1D signal.
    Args:
        x: Input tensor of shape [1, L].
        max_shift: Maximum absolute shift amount.
    Returns:
        shifted_x: Shifted signal of shape [1, L].
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    
    if shift == 0:
        return x
    
    if shift > 0:  # Left shift
        shifted_x = torch.cat([x[:, shift:], torch.zeros(1, shift)], dim=1)
    
    if shift < 0:  # Right shift
        shifted_x = torch.cat([torch.zeros(1, -shift), x[:, :shift]], dim=1)
    
    return shifted_x

def random_augment(x: torch.Tensor):
    """
    Apply random augmentation by combining zoom and shift.
    Args:
        x: Input tensor of shape [1, 256].
    Returns:
        augmented_x: Augmented signal of shape [1, 256].
    """
    x = random_zoom(x)
    x = random_shift(x)
    return x