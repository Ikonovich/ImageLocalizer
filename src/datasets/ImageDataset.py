import torch
import torchvision
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.utils import getImageTensor


# Takes a list of tuple(image paths, y Tensor).
# The y tensor can be None.
# During iteration, loads into memory and returns one image and the associated y at a time.
class ImageDataset:

    def __init__(self, data: list[tuple[str, Tensor | None]]):
        self.dataList = data
        self.curIndex = 0

    # Returns the next image in the iteration.
    def __next__(self):
        path, y = self.dataList[self.curIndex]

        imgTensor = self.__getitem__(self.curIndex)

        self.curIndex += 1
        if self.curIndex == len(self.dataList):
            self.curIndex = 0
        return imgTensor, torch.tensor(y)

    # Returns the element at the provided index.
    def __getitem__(self, index):
        path, y = self.dataList[index]

        imgTensor = getImageTensor(path=path, normalize=True)
        return imgTensor, torch.tensor(y, device="cuda")
