
import torch
import torch.nn.functional as F

def cosine_temperature_series(ttot, t_freq, T_high=9, T_low=7.0, device='cpu'):
    """
    Returns a tensor of length N = ttot / t_freq with one full cosine cycle:
    starts at T_high, dips to T_low mid-cycle, returns to T_high at the end.
    """
    N = int(round(ttot / float(t_freq)))
    if N < 2:
        raise ValueError("ttot/t_freq must be >= 2 to form a cosine cycle.")
    mean = 0.5 * (T_high + T_low)   # 8.0
    amp  = 0.5 * (T_high - T_low)   # 1.0
    theta = torch.linspace(0, 2*torch.pi, steps=N, device=device)  # inclusive of endpoints
    return mean + amp * torch.cos(theta)


def flat_then_linear_series(
    ttot,
    t_freq,
    flat_value=8.0,
    linear_start=None,
    slope=2.0,
    device='cuda:1'
):
    """
    Returns a tensor of length N = ttot / t_freq:
      - First half: constant at `flat_value`
      - Second half: linear ramp defined on x in [0, 1]:
            y(x) = linear_start + slope * x
        By default, linear_start == flat_value (continuous at midpoint).

    Parameters
    ----------
    ttot : float
        Total duration.
    t_freq : float
        Sampling interval.
    flat_value : float, optional
        Constant value for the first half.
    linear_start : float or None, optional
        Value at the midpoint (start of second half). If None, uses flat_value.
    slope : float, optional
        Total rise over the second half (since x ∈ [0,1], end = linear_start + slope).
    device : str, optional
        Torch device.

    Raises
    ------
    ValueError
        If N < 2.
    """
    N = int(round(ttot / float(t_freq)))
    if N < 2:
        raise ValueError("ttot/t_freq must be >= 2.")
    if linear_start is None:
        linear_start = float(flat_value)

    half = N // 2                           # first half length
    n2 = N - half                           # second half length
    first_half = torch.full((half,), float(flat_value), device=device)
    x = torch.linspace(0.0, 1.0, steps=n2, device=device)
    second_half = linear_start + slope * x

    return torch.cat([first_half, second_half])


def flat_then_gelu_series(
    ttot,
    t_freq,
    flat_value=8.0,
    linear_start=None,
    slope=1.0,
    sharpness=2.0,
    device='cpu'
):
    """
    Returns a tensor of length N = round(ttot / t_freq):
      - First half: constant at `flat_value`
      - Second half: smooth S-curve ramp (GELU-shaped) from 0 to 1, then scaled:
            y(x) = linear_start + slope * S_gelu(x; sharpness),
        where x ∈ [0,1] over the second half, and S_gelu maps [0,1] → [0,1] smoothly.

    Parameters
    ----------
    ttot : float
        Total duration.
    t_freq : float
        Sampling interval.
    flat_value : float, optional
        Constant value for the first half.
    linear_start : float or None, optional
        Value at the midpoint (start of the second half). If None, uses flat_value
        (i.e., continuous at the midpoint).
    slope : float, optional
        Total rise over the second half (end value = linear_start + slope).
    sharpness : float, optional
        Controls how quickly the ramp leaves zero and saturates near one.
        Larger => steeper mid-transition (default 2.0).
    device : str or torch.device, optional
        Torch device for the output tensors.

    Returns
    -------
    y : torch.FloatTensor
        Shape (N,). First half flat, second half GELU-smoothed ramp.
    """
    N = int(round(ttot / float(t_freq)))
    if N < 2:
        raise ValueError("ttot/t_freq must be >= 2.")
    if linear_start is None:
        linear_start = float(flat_value)

    # Split sizes
    half = N //3
    n2 = N - half

    # First half: constant
    first_half = torch.full((half,), float(flat_value), device=device, dtype=torch.float32)

    # Second half: x in [0,1]
    x = torch.linspace(0.0, 1.0, steps=n2, device=device, dtype=torch.float32)

    # Map x∈[0,1] to a GELU-shaped smoothstep in [0,1].
    # We use a centered input z = sharpness*(2x - 1), then affine-rescale GELU to [0,1]:
    #   S(x) = (gelu(z) - gelu(-sharpness)) / (gelu(sharpness) - gelu(-sharpness))
    z = sharpness * (2.0 * x - 1.0)
    g_pos = F.gelu(torch.tensor(sharpness, device=device))
    g_neg = F.gelu(torch.tensor(-sharpness, device=device))
    denom = (g_pos - g_neg)
    # Numerical safety (denom is positive for sharpness>0)
    if torch.abs(denom) < 1e-12:
        # Fall back to linear if sharpness is extremely tiny
        S = x
    else:
        S = (F.gelu(z) - g_neg) / denom

    second_half = linear_start + slope * S

    return torch.cat([first_half, second_half])
