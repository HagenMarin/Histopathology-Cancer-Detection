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