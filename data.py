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
from typing import Dict
import random
from torch import Tensor
import torchvision.transforms as tt
from astropy.visualization import HistEqStretch, ImageNormalize
from datetime import timedelta

scrap_date_list = [
    (datetime(1998, 5, 6), datetime(1998, 5, 7)),  # Solar Storm of May 1998
    (
        datetime(2000, 7, 14),
        datetime(2000, 7, 15),
    ),  # Bastille Day Solar Storm (July 2000)
    (datetime(2001, 4, 2), datetime(2001, 4, 3)),  # Solar Storm of April 2001
    (datetime(2001, 4, 6), datetime(2001, 4, 10)),  # Solar Storm of April 2001 (Series)
    (datetime(2001, 9, 25), datetime(2001, 9, 30)),  # Solar Storm of September 2001
    (
        datetime(2003, 10, 20),
        datetime(2003, 11, 2),
    ),  # Halloween Solar Storms (October-November 2003)
    (datetime(2005, 1, 15), datetime(2005, 1, 20)),  # Solar Storm of January 2005
    (datetime(2005, 7, 12), datetime(2005, 7, 16)),  # Solar Storm of July 2005
    (datetime(2008, 1, 24), datetime(2008, 1, 25)),  # Solar Storm of January 2008
    (datetime(2008, 3, 8), datetime(2008, 3, 10)),  # Solar Storm of March 2008
    (datetime(2010, 8, 1), datetime(2010, 8, 2)),  # Solar Storm of August 2010
    (datetime(2012, 3, 7), datetime(2012, 3, 8)),  # Solar Storm of March 2012
    (datetime(2012, 7, 23), datetime(2012, 7, 24)),  # Solar Storm of 2012
    (
        datetime(2014, 10, 22),
        datetime(2014, 10, 31),
    ),  # Halloween Solar Storms (October 2014)
    (datetime(2017, 9, 6), datetime(2017, 9, 7)),  # Solar Storm of September 2017
    (
        datetime(2017, 9, 10),
        datetime(2017, 9, 11),
    ),  # Solar Storm of September 2017 (Series)
    (
        datetime(2017, 9, 15),
        datetime(2017, 9, 16),
    ),  # Solar Storm of September 2017 (Series)
    (datetime(2021, 5, 26), datetime(2021, 5, 27)),  # Solar Storm of May 2021
]


def create_mask():
    # Create a rectangle
    n = random.randint(1, 31)  # Random integer for n
    m = random.randint(1, 31)  # Random integer for m
    left = random.randint(0, 32 - n) * 32  # Ensure multiple of 32
    top = random.randint(0, 32 - m) * 32  # Ensure multiple of 32
    width = n * 32
    height = m * 32

    mask = torch.ones(1024, 1024)
    mask[top:top+height, left:left+width] = 0

    return mask

class NormalizeInverse(tt.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


@dataclass
class CoronagraphDataset(Dataset):
    tool: str

    def __post_init__(self):
        self.path: Callable[[str], str] = lambda filename: os.path.join(
            Path(__file__).parent, "data", filename
        )
        self.image_paths: List[str] = [
            self.path(filename) for filename in sorted(os.listdir(self.path("")))
        ]
        # main transform
        self.transform = tt.Compose(
            [
                tt.Lambda(lambda x: ImageNormalize(stretch = HistEqStretch(x[np.isfinite(x)]))(x)),
                tt.ToTensor(),
                tt.Resize((1024, 1024), antialias=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img: np.array = fits.getdata(self.image_paths[idx]).astype(np.float32)

        img: Tensor = self.transform(img)

        mask = create_mask().unsqueeze(0)

        return img.type(torch.float32), mask

class CrossDataset(CoronagraphDataset):
    def __init__(self, tool):
        super().__init__(tool)
    def __len__(self) -> int:
        return super().__len__() - 1
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        x1_path = self.image_paths[idx]
        x2_path = self.image_path[idx + 1]
        x_1: np.array = fits.getdata(x1_path).astype(np.float32)
        time_1: float = datetime(fits.getheader(x1_path)['date_obs'].split('T')[-1], '%H:%m:%S.%f')
        x_2: np.array = fits.getdata(x2_path).astype(np.float32)
        time_2: float = datetime(fits.getheader(x2_path)['date_obs'].split('T')[-1], '%H:%m:%S.%f')

        x_1: Tensor = self.transform(x_1)
        x_2: Tensor = self.transform(x_2)

        time = ((time_2 - time_1)/timedelta(minutes = 24)).total_seconds()/60

        mask = create_mask().unsqueeze(0)

        return x_1, x_2, time, mask
@dataclass
class Data:
    tool: str

    def __post_init__(self):
        self.path: str = os.path.join(Path(__file__).parent, "data")
        self.__downloader = downloader(self.tool, self.path)

    async def __call__(self, scrap_date_list):
        await self.__downloader(scrap_date_list)

    def level_1(self):
        paths = [os.path.join(self.path, self.tool, filename) for filename in sorted(os.listdir(os.path.join(self.path, self.tool)))]
        level_1(
            paths,
            self.path,
            'fits'
        )
            

if __name__ == "__main__":
    downloader = Data("c3")
    asyncio.run(downloader(scrap_date_list))
    downloader.level_1()
