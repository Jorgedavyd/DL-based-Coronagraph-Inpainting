from corkit.lasco import level_1, downloader
from torch.utils.data import Dataset
from pathlib import Path
import os
from datetime import datetime
import asyncio
import torch
from astropy.io import fits

scrap_date_list = [
    (datetime(2003, 10, 20), datetime(2003, 10, 30))
]

def create_mask(x, y):
    tensor = torch.ones(x, y)
    # Generate random coordinates for the top-left corner of the square
    top_left_x = torch.randint(0, x - 32, (1,))
    top_left_y = torch.randint(0, y - 32, (1,))
    
    # Apply the mask
    tensor[top_left_x:top_left_x + 32, top_left_y:top_left_y + 32] = 0
    
    return tensor

class CoronagraphDataset(Dataset):
    def __init__(self, tool):
        self.path = lambda filename: os.path.join(Path(__file__).parent, 'data', tool, filename)
        self.image_paths = [self.path(filename) for filename in os.listdir(self.path(''))]
        
    def get_ind(self, path):
        img = fits.getdata(path)
        return torch.from_numpy(img)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = self.get_ind(self.image_paths[idx])
        mask = create_mask()
        return {
            'img': img,
            'mask': mask
        }
    
class Data:
    def __init__(
            self,
            tool,
    ):
        self.path = os.path.join(Path(__file__).parent, 'data')
        self.tool = tool
        self.downloader = downloader(tool, self.path)
    async def __call__(self, scrap_date_list):
        await self.downloader(scrap_date_list)
    def level_1(self):
        for filename in os.listdir(os.path.join(self.path, self.tool)):
            path = os.path.join(self.path, self.tool, filename)
            level_1(
                path,
                path
            )

if __name__ == '__main__':
    downloader = Data('c2')
    asyncio.run(downloader(scrap_date_list))
    downloader.level_1()
    