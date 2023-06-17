import torch
import torch.nn as nn
class model_class( nn.Module ):
    
    def __init__(self): 
        super(model_class, self).__init__()
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