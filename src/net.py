from torch import nn

class Net(nn.Module):
    def __init__(self, input_size: int, num_classes: int, dimension: int):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, dimension),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(dimension, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x