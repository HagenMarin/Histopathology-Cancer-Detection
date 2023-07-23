import torch
import architecture
import os
import cv2
from pathlib import Path
import glob
from data_loading import iterate_batches

model = architecture.LeNet_kaiming_normal()
model.load_state_dict(torch.load('model/LeNet_kn_ta_0.8631193181818182_va_0.7842565597667639.pickle'))
cwd = Path.cwd()
dirlist = sorted(glob.glob('train/*.tif'))
infile = dirlist[0]
print(infile)
filename = os.path.join(cwd, infile)
image = cv2.imread(filename)
img = image.astype(float)/255
img = img.reshape((3,96,96))
batch = torch.empty((1,3,96,96),dtype=torch.float32)
batch[0] = torch.tensor(img,dtype=torch.float64)
print(model(batch)[0])

