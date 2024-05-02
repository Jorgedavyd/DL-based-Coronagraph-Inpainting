import torch
import os

## GPU usage


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def create_config(name_run: str):
    os.makedirs(f"./{name_run}/models", exist_ok=True)
    return {
        "name": name_run,
        "device": get_default_device(),
        "epoch": 0,
        "global_step_train": 0,
        "global_step_val": 0,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
    }


from torch import nn
from torch import Tensor


class FourierMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fft_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fft = torch.fft.fftn

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        fft_x = self.fft(x, dim=(-1, -2))
        x = x.view(b, c, -1)
        out, _ = self.attention(x, x, x)
        out_fft, _ = self.fft_attention(
            fft_x.real.view(b, c, -1), fft_x.imag.view(b, c, -1), x
        )
        return (out + out_fft).view(b, c, h, w)
