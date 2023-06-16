from pathlib import Path
import glob
import torch
import os
import cv2
import pandas as pd

def loadDataframe():
    cwd = Path.cwd()
    filename = os.path.join(cwd, 'train_labels.csv')
    df = pd.read_csv(filename)
    df = df.sort_values('id')
    return df


def dataIterator(df, batch_size):
    #print(df['23e49215068a2bc642fae1cc75cac1e2ea926314'])
    cwd = Path.cwd()
    df = df.to_numpy()
    count = 0
    dirlist = sorted(glob.glob('train/*.tif'))
    
    for i in range(0,df.shape[0]-batch_size,batch_size):
        batch_imgs = torch.empty((batch_size,3,96,96),dtype=torch.float32)
        batch_labels = torch.empty((batch_size,1),dtype=torch.float32)
        for n in range(batch_size):
            #print("Current File Being Processed is: " + str(i+n))
            infile = dirlist[i+n]
            filename = os.path.join(cwd, infile)
            image = cv2.imread(filename)
            img = image.astype(float)/255
            img = img.reshape((3,96,96))
            #print(img)
            #print(img.shape)
            batch_imgs[n]=torch.tensor(img,dtype=torch.float64)
            batch_labels[n]=torch.tensor(float(df[i+n,1]),dtype=torch.float64)
        yield (batch_imgs,batch_labels)
'''
class preloadDataIterator():
    
    def __init__(self, df, batch_size):
        self.cwd = Path.cwd()
        self.df = df.to_numpy()
        self.count = 0
        self.num_splits = 4
        self.batch_size = batch_size
'''        


def dataIteratorPreload(df, batch_size):
    #print(df['23e49215068a2bc642fae1cc75cac1e2ea926314'])
    cwd = Path.cwd()
    df = df.to_numpy()
    count = 0
    num_splits = 4

    
    dirlist = sorted(glob.glob('train/*.tif'))
    splitsize = int(len(dirlist)/num_splits)
    imgs = torch.empty((int(len(dirlist)/num_splits),3,96,96),dtype=torch.float32)
    for split in range(num_splits):
        for i in range(split*splitsize,(split+1)*splitsize):
            infile = dirlist[i]
            filename = os.path.join(cwd, infile)
            image = cv2.imread(filename)
            img = image.astype(float)/255
            img = img.reshape((3,96,96))
            #print(img)
            #print(img.shape)
            imgs[i-split*splitsize]=torch.tensor(img,dtype=torch.float32)
        for i in range(0,splitsize-batch_size,batch_size):
            batch_imgs =torch.empty((batch_size,3,96,96),dtype=torch.float32)
            batch_labels = torch.empty((batch_size,1),dtype=torch.float32)
            for n in range(batch_size):
                #print("Current File Being Processed is: " + str(i+n))
                batch_imgs[n]=imgs[i+n]
                batch_labels[n]=torch.tensor(float(df[i+n,1]),dtype=torch.float64)
            yield (batch_imgs,batch_labels)

def makeTrainDataset(df, size, dirlist):
    datasetArr = []
    for i, instance in enumerate(dataIterator(df,1,dirlist)):
        datasetArr.append(instance[0])
        if(not i<size):
            break
    return datasetArr