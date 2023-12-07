import torch

from src.datasets.Process import loadTestset, loadSimpleSample
from src.model.LocalizerModel import Localizer


def main():
    model = Localizer()
    # dataset = loadTestset()
    dataset = loadSimpleSample()
    train(model, dataset)


def train(model, dataset):
    lossFunction = torch.nn.MSELoss()
    cumulativeLoss = 0.
    lastLoss = 0.
    learnRate = 0.05

    # How often it prints the latest result and resets the cumulative lost.
    printEvery = 20

    optimizer = torch.optim.SGD(model.parameters(), lr=learnRate, momentum=0.8)

    for i in range(1000):

        for j, data in enumerate(dataset):
            x, y = data

            optimizer.zero_grad()
            output = model(x)
            loss = lossFunction(output, y)
            loss.backward()

            optimizer.step()

            cumulativeLoss += loss.item()

            if (i * (j + 1)) % printEvery == (printEvery - 1):
                lastLoss = cumulativeLoss / printEvery  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, lastLoss))
                cumulativeLoss = 0


def test():
    pass


if __name__ == "__main__":
    main()
