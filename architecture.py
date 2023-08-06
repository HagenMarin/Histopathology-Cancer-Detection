import torch
import torch.nn as nn
from torchvision import transforms
from typing import Type

class LeNet_kaiming_normal( nn.Module ):
    
    def __init__(self): 
        super(LeNet_kaiming_normal, self).__init__()
        self.layers = nn.Sequential( 
            nn.Conv2d(3,6,5,padding=2),
            nn.ReLU(), 
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,5,padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Flatten(),
            nn.Linear(7744,120), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(120,84), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(84,1),
            nn.Sigmoid())  # we are predicting only two classes, so we can use one sigmoid neuron as output
        torch.nn.init.kaiming_normal_(self.layers[0].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[3].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[7].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[9].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[11].weight, mode='fan_in')

    def forward( self, x ): # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x
    
class LeNet( nn.Module ):
    
    def __init__(self): 
        super(LeNet, self).__init__()
        self.layers = nn.Sequential( 
            nn.Conv2d(3,6,5,padding=2),
            nn.ReLU(), 
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,5,padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Flatten(),
            nn.Linear(7744,120), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(120,84), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(84,1),
            nn.Sigmoid())  # we are predicting only two classes, so we can use one sigmoid neuron as output

    def forward( self, x ): # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x
    
class NiN( nn.Module ):
    
    def __init__(self):
        super(NiN,self).__init__()
        self.layers = nn.Sequential(nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1), nn.ReLU()),
            nn.MaxPool2d(3, stride=2),
            nn.Sequential(
            nn.Conv2d(96 ,256, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.ReLU()),
            nn.MaxPool2d(3, stride=2),
            nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1), nn.ReLU()),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nn.Sequential(
            nn.Conv2d(384, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1), nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1), nn.ReLU()),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid())
        torch.nn.init.kaiming_normal_(self.layers[0][0].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[0][2].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[0][4].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[2][0].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[2][2].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[2][4].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[4][0].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[4][2].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[4][4].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[7][0].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[7][2].weight, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layers[7][4].weight, mode='fan_in')
        
    def forward( self, x ): # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x

class DenseModel(nn.Module):
    def __init__(self) -> None: 
        super(DenseModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d((32,32)),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(3072),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(3072, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward( self, x )  -> torch.Tensor: # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x

class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs) -> torch.Tensor:
        return self.module(inputs) + inputs
    
class ResNet(nn.Module):
    def __init__(self) -> None: 
        super(ResNet, self).__init__()
        self.padding = 1
        self.layers = torch.nn.Sequential(
            # transforms.RandomRotation(180),
            # transforms.RandomHorizontalFlip(0.3),
            # transforms.ColorJitter(0.1,0.1,0.1,0.1),
            torch.nn.Conv2d(3, 32, kernel_size=4),
            # 32 filters in and out, no max pooling so the shapes can be added
            ResidualBlock(
                torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=3, padding=self.padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 32, kernel_size=3, padding=self.padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(32),
                )
            ),
            # Another ResidualBlock block, you could make more of them
            # Downsampling using maxpool and others could be done in between etc. etc.
            ResidualBlock(
                torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=3,padding=self.padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 32, kernel_size=3,padding=self.padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(32),
                )
            ),
            # ResidualBlock(
            #    torch.nn.Sequential(
            #        torch.nn.Conv2d(32, 32, kernel_size=3,padding=self.padding),
            #        torch.nn.ReLU(),
            #        torch.nn.BatchNorm2d(32),
            #        torch.nn.Conv2d(32, 32, kernel_size=3,padding=self.padding),
            #        torch.nn.ReLU(),
            #        torch.nn.BatchNorm2d(32),
            #    )
            # ),
            #nn.Dropout(0.2),
            # Pool all the 32 filters to 1, you may need to use `torch.squeeze after this layer`
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            # 32 1 class
            torch.nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward( self, x )  -> torch.Tensor: # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x


#ResNet18 implementation from here:https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

class ResNet18(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(ResNet18, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.transform1 = transforms.RandomRotation(180)
        self.transform2 = transforms.RandomHorizontalFlip(0.3)
        self.transform3 = transforms.ColorJitter(0.1,0.1,0.1,0.1)
        
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.dropout2 = nn.Dropout(0.7)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.dropout3 = nn.Dropout(0.6)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.bn2 = nn.BatchNorm2d(512*self.expansion)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512*self.expansion, num_classes)
        self.sig = nn.Sigmoid()
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.transform1(x)
            x = self.transform2(x)
            #x = self.transform3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        #x = self.dropout1(x)
        x = self.layer2(x)
        #x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        #x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sig(x)
        return x