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
import torchvision.transforms as tt
from astropy.visualization import HistEqStretch, ImageNormalize

scrap_date_list = [
    (datetime(1998, 5, 6), datetime(1998, 5, 7)),  # Solar Storm of May 1998
    (datetime(2000, 7, 14), datetime(2000, 7, 15)),  # Bastille Day Solar Storm (July 2000)
    (datetime(2001, 4, 2), datetime(2001, 4, 3)),  # Solar Storm of April 2001
    (datetime(2001, 4, 6), datetime(2001, 4, 10)),  # Solar Storm of April 2001 (Series)
    (datetime(2001, 9, 25), datetime(2001, 9, 30)),  # Solar Storm of September 2001
    (datetime(2003, 10, 20), datetime(2003, 11, 2)),  # Halloween Solar Storms (October-November 2003)
    (datetime(2005, 1, 15), datetime(2005, 1, 20)),  # Solar Storm of January 2005
    (datetime(2005, 7, 12), datetime(2005, 7, 16)),  # Solar Storm of July 2005
    (datetime(2008, 1, 24), datetime(2008, 1, 25)),  # Solar Storm of January 2008
    (datetime(2008, 3, 8), datetime(2008, 3, 10)),  # Solar Storm of March 2008
    (datetime(2010, 8, 1), datetime(2010, 8, 2)),  # Solar Storm of August 2010
    (datetime(2012, 3, 7), datetime(2012, 3, 8)),  # Solar Storm of March 2012
    (datetime(2012, 7, 23), datetime(2012, 7, 24)),  # Solar Storm of 2012
    (datetime(2014, 10, 22), datetime(2014, 10, 31)),  # Halloween Solar Storms (October 2014)
    (datetime(2017, 9, 6), datetime(2017, 9, 7)),  # Solar Storm of September 2017
    (datetime(2017, 9, 10), datetime(2017, 9, 11)),  # Solar Storm of September 2017 (Series)
    (datetime(2017, 9, 15), datetime(2017, 9, 16)),  # Solar Storm of September 2017 (Series)
    (datetime(2021, 5, 26), datetime(2021, 5, 27)),  # Solar Storm of May 2021
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
        ## Computing the mean and standard deviation
        
        
        #main transform
        self.transform = tt.Compose([
            tt.Lambda(lambda img0: ImageNormalize(stretch=HistEqStretch(img0[np.isfinite(img0)]))(img0)),
            tt.ToTensor(),
            tt.Resize((1024,1024), antialias=True)
        ])
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img: np.array = fits.getdata(self.image_paths[idx]).astype(np.float32)

        img: Tensor = self.transform(img)

        mask = create_mask().unsqueeze(0)
        
        return img.type(torch.float32), mask

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
    