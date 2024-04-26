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


def import_config(name_run):
    try:
        files = os.listdir(f"./{name_run}/models")
        if len(files) == 0:
            raise FileNotFoundError
        for idx, file in enumerate(files):
            print(f"{idx+1}. {file}")
        config = int(input("Choose the config: "))
        return torch.load(files[config - 1])
    except FileNotFoundError:
        return create_config(name_run)
