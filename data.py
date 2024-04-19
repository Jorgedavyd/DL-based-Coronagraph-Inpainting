from corkit.lasco import level_1, downloader
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Callable, List
from datetime import datetime
from astropy.io import fits
from pathlib import Path
import asyncio
import torch
import os
import numpy as np
from typing import Dict, Union
from torchvision.transforms import Resize, Normalize
from torch import Tensor
import matplotlib.pyplot as plt
from corkit.lasco import LASCOplot, Plot

scrap_date_list = [
    (datetime(2003, 10, 20), datetime(2003, 10, 30))
]

def create_mask(x: int = 1024, y: int = 1024):
    tensor = torch.ones((x, y))
    # Generate random coordinates for the top-left corner of the square
    top_left_x = torch.randint(0, x - 32, (1,))
    top_left_y = torch.randint(0, y - 32, (1,))
    
    # Apply the mask
    tensor[top_left_x:top_left_x + 32, top_left_y:top_left_y + 32] = 0
    
    return tensor

@dataclass
class CoronagraphDataset(Dataset):
    tool: str

    def __post_init__(self):
        self.path: Callable[[str], str] = lambda filename: os.path.join(Path(__file__).parent, 'data', self.tool, filename)
        self.image_paths: List[str] = [self.path(filename) for filename in os.listdir(self.path(''))]
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img: Tensor = fits.getdata(self.image_paths[idx]).astype(np.float64)
        if img.shape != 2:
            raise ValueError('bro...')
        img: Tensor = torch.from_numpy(img)
        img: Tensor = Resize((1024, 1024), antialias=True)(img)
        img: Tensor = Normalize(0, 1)(img)

        mask = create_mask()

        return img.unsqueeze(0), mask.unsqueeze(0)

@dataclass
class Data:
    tool: str
    def __post_init__(self):
        self.path: str = os.path.join(Path(__file__).parent, 'data')
        self.__downloader = downloader(self.tool, self.path)
    async def __call__(self, scrap_date_list):
        await self.__downloader(scrap_date_list)
    def level_1(self):
        for filename in os.listdir(os.path.join(self.path, self.tool)):
            path = os.path.join(self.path, self.tool, filename)
            level_1(path, path)

if __name__ == '__main__':
    downloader = Data('c2')
    asyncio.run(downloader(scrap_date_list))
    downloader.level_1()
    