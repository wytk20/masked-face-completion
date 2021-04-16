
import torch
import torch.nn as nn
from torchvision import models

class FaceRecov_Net(nn.Module):
    def __init__(self):
        super(FaceRecov_Net, self).__init__()

        resnet_ft = models.resnext50_32x4d(pretrained=True)
        #num_ftrs = resnet_ft.fc.in_features
        #resnet_ft.fc = nn.Linear(num_ftrs, num_classes)
        #resnet_ft.load_state_dict(torch.load('../../checkpoint/webface_501_pretrained.pth'))

        self.head_layer = nn.Sequential(
            resnet_ft.conv1, resnet_ft.bn1, resnet_ft.relu,
            resnet_ft.maxpool)
        self.layer1 = resnet_ft.layer1
        self.layer2 = resnet_ft.layer2
        self.layer3 = resnet_ft.layer3
        self.layer4 = resnet_ft.layer4

        self.uplayer4 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(True))

        self.uplayer3 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True))

        self.uplayer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))

        self.uplayer1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))

        self.convT = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.out = nn.ReLU(True)

    def forward(self, input):
        x = self.head_layer(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.uplayer4(x)
        x = self.uplayer3(x)
        x = self.uplayer2(x)
        x = self.uplayer1(x)

        x = self.convT(x)
        x = self.out(x)

        return x