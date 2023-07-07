import torch
import torch.nn as nn

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
    
class ResidualBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
    
class ResNet(nn.Module):
    def __init__(self): 
        super(ResNet, self).__init__()
        self.padding = 1
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7),
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
            # Pool all the 32 filters to 1, you may need to use `torch.squeeze after this layer`
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            # 32 1 class
            torch.nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward( self, x ): # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x