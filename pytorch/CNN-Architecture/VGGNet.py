import torch.nn as nn

VGG_TYPES = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.con_layers = self.create_conv_layer(VGG_TYPES["VGG19"])

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # image size= 224/2^5=7
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels
        for i in architecture:
            if type(i) == int:
                out_channels = i
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(i),
                    nn.ReLU(),
                ]

                in_channels = i

            elif i == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)