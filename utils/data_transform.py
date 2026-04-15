"""
Data transforms: reshape IMU time series into formats suitable for SNN models.
"""
import torch
import numpy as np
from typing import Tuple


def imu_to_image(imu_data: torch.Tensor, img_size: int = 32) -> torch.Tensor:
    """
    Map IMU time series to a pseudo-image tensor.

    Args:
        imu_data: Shape (B, T, C) or (B, C, T).
        img_size: Target spatial size (H=W).

    Returns:
        Tensor of shape (B, C, H, W).
    """
    if len(imu_data.shape) == 3:
        if imu_data.shape[1] == 3:  # (B, C, T)
            B, C, T = imu_data.shape
        else:  # (B, T, C)
            B, T, C = imu_data.shape
            imu_data = imu_data.transpose(1, 2)  # (B, C, T)
    else:
        raise ValueError(f"Unsupported input shape: {imu_data.shape}")

    B, C, T = imu_data.shape

    # Option A: reshape when T is near img_size^2
    if T >= img_size * img_size:
        target_len = img_size * img_size
        if T > target_len:
            imu_data = imu_data[:, :, :target_len]
        else:
            padding = torch.zeros(B, C, target_len - T, device=imu_data.device)
            imu_data = torch.cat([imu_data, padding], dim=2)

        images = imu_data.reshape(B, C, img_size, img_size)
    else:
        # Option B: pad / interpolate
        h = int(np.sqrt(T))
        w = T // h
        if h * w < T:
            w += 1

        target_len = h * w
        if T < target_len:
            padding = torch.zeros(B, C, target_len - T, device=imu_data.device)
            imu_data = torch.cat([imu_data, padding], dim=2)

        images = imu_data[:, :, :h*w].reshape(B, C, h, w)

        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size), mode='bilinear', align_corners=False
        )

    return images


def imu_to_spectrogram(imu_data: torch.Tensor, img_size: int = 32) -> torch.Tensor:
    """
    Map IMU time series to a magnitude spectrogram-like tensor.

    Args:
        imu_data: Shape (B, T, C) or (B, C, T).
        img_size: Target spatial size.

    Returns:
        Tensor of shape (B, C, H, W).
    """
    import torch.fft

    if len(imu_data.shape) == 3:
        if imu_data.shape[1] == 3:  # (B, C, T)
            B, C, T = imu_data.shape
        else:  # (B, T, C)
            B, T, C = imu_data.shape
            imu_data = imu_data.transpose(1, 2)  # (B, C, T)
    else:
        raise ValueError(f"Unsupported input shape: {imu_data.shape}")

    B, C, T = imu_data.shape

    fft_data = torch.fft.rfft(imu_data, dim=2)
    magnitude = torch.abs(fft_data)

    magnitude = torch.nn.functional.interpolate(
        magnitude.unsqueeze(0), size=(img_size, img_size),
        mode='bilinear', align_corners=False
    ).squeeze(0)

    return magnitude


def prepare_snn_input(data: torch.Tensor, time_steps: int = 4) -> torch.Tensor:
    """
    Expand static (B, C, H, W) into SNN-style (T, B, C, H, W) by repeating along time.

    Args:
        data: Shape (B, C, H, W).
        time_steps: Number of time steps T.

    Returns:
        Tensor of shape (T, B, C, H, W).
    """
    if len(data.shape) == 4:
        B, C, H, W = data.shape
        data = data.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
        return data
    else:
        raise ValueError(f"Unsupported input shape: {data.shape}")
