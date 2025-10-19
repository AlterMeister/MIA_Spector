'''
    Analysis for the text LVLMs

    We only use the fourier lowpass feature
'''

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class Perturb:
    def __init__(self, fourier_low : float = 0.3) -> None:
        self.fourier_low = fourier_low

    # lowpass fourier perturb
    def fourier_lowpass(self, img_pil_or_path, cutoff_ratio=0.2, per_channel=True):
        img = self.ensure_pil(img_or_path=img_pil_or_path)
        arr = self._to_float01(img)

        H, W, C = arr.shape
        R = int(min(H, W) * 0.5 * cutoff_ratio)

        out = np.zeros_like(arr)

        for ch in range(C if per_channel else 1):
            channel = arr[..., ch] if per_channel else (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
            Fshift = np.fft.fftshift(np.fft.fft2(channel))
            mask = self._make_circular_mask(H, W, radius=R)
            Fshift_masked = Fshift * mask
            rec = np.fft.ifft2(np.fft.ifftshift(Fshift_masked)).real
            if per_channel:
                out[..., ch] = rec
            else:
                out[..., 0] = rec
                out[..., 1] = rec
                out[..., 2] = rec
        
        out = np.clip(out, 0.0, 1.0)
        return self._from_float01(out)

    @staticmethod
    def ensure_pil(img_or_path):
        if isinstance(img_or_path, str):
            return Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            return img_or_path.convert("RGB")
        else:
            raise ValueError("img_or_path must be filepath or PIL.Image")
    
    @staticmethod
    def _to_float01(img_pil):
        arr = np.array(img_pil).astype(np.float32) / 255.0
        return arr

    @staticmethod
    def _from_float01(arr):
        arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_u8)

    @staticmethod
    def _fft2_channel(channel):
        return np.fft.fft2(channel), np.fft.fftshift(np.fft.fft2(channel))

    @staticmethod
    def _make_circular_mask(H, W, center=None, radius=None):
        if center is None:
            center = (int(W/2), int(H/2))
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        if radius is None:
            raise ValueError("radius must be provided")
        mask = dist <= radius
        return mask
    