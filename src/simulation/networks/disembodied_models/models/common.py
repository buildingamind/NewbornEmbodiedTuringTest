import torch.nn as nn

def create_encoder(channels):
    # Create convolutional encoder.
    in_channels = channels[0]
    encoder_modules = []
    for out_channels in channels[1:]:
        encoder_modules.extend([
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU()
        ])
        in_channels = out_channels
    encoder_modules.append(nn.Flatten())
    encoder = nn.Sequential(*encoder_modules)
    return encoder


def create_decoder(channels):
    # Create convolutional decoder.
    in_channels = channels[0]
    decoder_modules = []
    for out_channels in channels[1:]:
        decoder_modules.extend([
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU()
        ])
        in_channels = out_channels
    decoder = nn.Sequential(*decoder_modules)
    return decoder


class AlexNet(nn.Module):
    def __init__(self, height, width, input_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim # 512
        # self.conv_operation = nn.Sequential(
        #     # unflatten from [128,512] -> [128,512,1,1], assuming the output from the last block is 1X1
        #     nn.Unflatten(1, (512, 1, 1)),
        #     nn.Conv2d(512, 1024, kernel_size=(3,3), stride=1, padding=1, bias=False), #in_channels, out_channels , kernel_size, stride, padding
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(1024),
        #     nn.Conv2d(1024, 2048, kernel_size=(3,3), stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(2048),
        #     nn.Flatten(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(2048, 1024, bias=True),
        #     nn.Linear(1024, 512, bias=True),
        #     nn.Linear(512, 1, bias=True),
        # ) -> correct one
        # self.conv_operation = nn.Sequential(
        #     # unflatten from [128,512] -> [128,512,1,1], assuming the output from the last block is 1X1
        #     nn.Unflatten(1, (32, 4, 4)),
        #     nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1, bias=False), #in_channels, out_channels , kernel_size, stride, padding
        #     nn.ReLU(inplace=True),
        #     # nn.BatchNorm2d(64),
        #     # nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1, bias=False),
        #     # nn.ReLU(inplace=True),
        #     # nn.BatchNorm2d(128),
        #     nn.Flatten(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 512, bias=True),
        #     nn.Linear(512, 1, bias=True),
        # )

        self.block = nn.Sequential(
            
            nn.Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 1, bias=True)
        )

    def forward(self, x):
        """
            1. check shape before and after flattening
            2. check the shape from the resnet last block
            3. check ae encoder - decoder functionality
        """

        #print("before flatten - ", x.shape) #[128,512]
        #logits = self.conv_operation(x) # [128,1024]
        #print("size - ", logits.shape)
        #print("value after conv operation and flattening it - ", conved_val.shape) 
        #print("after unflattening - ", unflattened_val.shape) # [128, 512, 1, 1]
        logits = self.block(x)

        return logits


class LinearProbeMultiClass(nn.Module):

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim # 512
        self.block = nn.Sequential(
            
            nn.Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 10, bias=True)
        )

    def forward(self, x):
        logits = self.block(x)
        return logits
