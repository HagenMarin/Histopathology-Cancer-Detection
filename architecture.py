import torch
import torch.nn as nn
class model_class( nn.Module ):
    
    def __init__(self): 
        super(model_class, self).__init__()
        self.layers = nn.Sequential( 
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(), 
            nn.Conv2d(64,16,3,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(147456,256), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(256,128), # the 128 here is the number of elements in the tensor computed so far
            nn.ReLU(), 
            nn.Linear(128,1),
            nn.Sigmoid())  # we are predicting only two classes, so we can use one sigmoid neuron as output
    
    def forward( self, x ): # computes the forward pass ... this one is particularly simple
        x = self.layers( x )
        return x