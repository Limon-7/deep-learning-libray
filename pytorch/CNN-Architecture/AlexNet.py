import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fc1 = nn.Linear(256* 6* 6, 4096) # flatten layer
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))

        x = self.pool(x)

        x = self.relu(
            self.conv3(x)
        )
        x = self.relu(
            self.conv4(x)
        )
        x = self.relu(
            self.conv5(x)
        )
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x=self.fc1(x)

        x = self.fc2(x)
        x = self.fc3(x)
        return x
