import torch
import architecture
import os
import cv2
from pathlib import Path
import glob
from data_loading import iterate_batches
from data_loading import get_combined_dirlist
from data_loading import loadDataframe
from data_loading import dataIterator
from data_loading import dataIterator_ids

model = architecture.ResNet()
model.load_state_dict(torch.load('model/ResNet_ta_0.910958904109589_va_0.8849957191780822.pickle'))
df = loadDataframe()
trainloader = dataIterator_ids(df,128, 0,0.7,0.15)
for batch, labels, ids in trainloader:
    outputs = model(batch)
    for i, ouput in enumerate(outputs):
        if (ouput>0.5) != labels[i]:
            print('output:' + str(ouput))
            print('label:' + str(labels[i]))
            print(ids[i])
    #print(>0.5)
    print(sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item())
    quit()




