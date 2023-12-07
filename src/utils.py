import json

import torch
import torchvision


# Reads in the image at the provided path and returns it.
def getImageTensor(path: str, normalize: bool = True):
    imgTensor = torchvision.io.read_image(path)

    if normalize:
        imgTensor = torch.div(imgTensor, 255)
    return imgTensor.to(device="cuda")


# Write an object to a json file.
def writeJson(data, path):
    jsonDict = json.dumps(obj=data)

    with open(path, "w") as outfile:
        outfile.write(jsonDict)


# Read from a json-formatted text file.
def readJson(path):
    with open(path, "r") as infile:
        data = infile.read()

    return json.loads(data)
