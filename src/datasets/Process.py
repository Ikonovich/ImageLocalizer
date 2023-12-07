import os
from pathlib import Path

from src.datasets.ImageDataset import ImageDataset
from src.utils import readJson, writeJson
import xml.etree.ElementTree as ET

import torch

# Used to process datasets to make them compatible with ImageDataset.
# ImageDataset expects a list of tuples containing an image filepath and the
# corresponding y values.

ROOT_PATH = str(Path(__file__).parents[2])
DATASETS = os.path.join(ROOT_PATH, "Datasets")


# Load the simple sample.
def loadSimpleSample():
    imageData = os.path.join(DATASETS, "test_sample.ds")
    data = readJson(imageData)
    return ImageDataset(data)

# Processes the cucumber/mushroom test dataset.
def loadTestset():
    rootPath = os.path.join(DATASETS, "Testset")
    filepaths = os.listdir(rootPath)
    processedPath = os.path.join(rootPath, "Processed.ds")
    if processedPath in filepaths:
        data = readJson(processedPath)
        return ImageDataset(data)

    images = list()
    for path in filepaths:
        if ".xml" in path:
            xml = ET.parse(os.path.join(rootPath, path))
            root = xml.getroot()
            # Get the image filename from the XML file and create the full path to it.
            filename = root.find("filename").text
            imagePath = os.path.join(rootPath, filename)
            bounds = getBoundingBox(root)
            images.append((imagePath, bounds))

    writeJson(images, processedPath)
    return ImageDataset(images)


# Returns the bounding attributes in the provided XML root.
def getBoundingBox(root) -> list[float]:
    obj = root.find("object")
    bndbox = obj.find("bndbox")
    xmax = float(bndbox.find("xmax").text)
    xmin = float(bndbox.find("xmin").text)
    ymax = float(bndbox.find("ymax").text)
    ymin = float(bndbox.find("ymin").text)
    return [xmin, xmax, ymin, ymax]


if __name__ == "__main__":
    loadTestset()
