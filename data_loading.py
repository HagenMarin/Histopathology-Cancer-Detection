from pathlib import Path
import glob
import torch
import os
import cv2
import pandas as pd
import pickle
import random

#load the provided csv into a Pandas Dataframe and sort it by the id, to later match the images more easily
def loadDataframe(dir = ''):
    if dir == '':
        cwd = Path.cwd()
        filename = os.path.join(cwd, 'train_labels.csv')
    else:
        filename = dir + 'train_labels.csv'
    df = pd.read_csv(filename)
    df = df.sort_values('id')
    return df

#creates the individual batches and stores them in 3 seperate paths for training, validation and testing
def create_batches(batch_size):
    cwd = Path.cwd()
    
    dataframe = loadDataframe()
    print('generating train batches')
    trainloader = dataIterator(dataframe,batch_size, 0,0.7,0.15)
    for i, minibatch in enumerate(trainloader,0):
        filename = str(cwd)+ '/train_batches'+str(batch_size)+'/batch_'+str(i)+'.pickle'
        print(filename)
        with open(filename, 'wb+') as handle:
            pickle.dump(minibatch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('generating valid batches')
    trainloader = dataIterator(dataframe,batch_size, 1,0.7,0.15)
    for i, minibatch in enumerate(trainloader,0):
        filename = str(cwd)+ '/valid_batches'+str(batch_size)+'/batch_'+str(i)+'.pickle'
        print(filename)
        with open(filename, 'wb+') as handle:
            pickle.dump(minibatch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('generating test batches')
    trainloader = dataIterator(dataframe,batch_size, 2,0.7,0.15)
    for i, minibatch in enumerate(trainloader,0):
        filename = str(cwd)+ '/test_batches'+str(batch_size)+'/batch_'+str(i)+'.pickle'
        print(filename)
        with open(filename, 'wb+') as handle:
            pickle.dump(minibatch, handle, protocol=pickle.HIGHEST_PROTOCOL)

#iterates over the premade test batches by loading them from storage one by one
def iterate_test_batches(batch_size = 128, dir = ''):
    dirlist = get_test_dirlist(batch_size, dir)
    for i, filename in enumerate(dirlist):
        with open(filename, 'rb') as handle:
            minibatch = pickle.load(handle)
        yield minibatch

#iterates over the premade batches by loading them from storage one by one
def iterate_batches(batch_size = 128, train=True, shuffle=True):
    if train:
        dirlist = get_train_dirlist(batch_size)
    else:
        dirlist = get_valid_dirlist(batch_size)
    if shuffle:
        random.shuffle(dirlist)
    for i, filename in enumerate(dirlist):
        with open(filename, 'rb') as handle:
            minibatch = pickle.load(handle)
        yield minibatch

def iterate_batches_kfold(kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True):
    dirlist = get_combined_dirlist(batch_size)
    #print(dirlist)
    vallist = dirlist[(currentfold)*int(len(dirlist)/kfolds):(currentfold+1)*int(len(dirlist)/kfolds)]
    trainlist = [p for p in dirlist if p not in vallist]
    #print(trainlist)
    usedlist = trainlist
    if shuffle:
        random.shuffle(trainlist)
    if not train:
        usedlist = vallist
    for filename in usedlist:
        with open(filename, 'rb') as handle:
            minibatch = pickle.load(handle)
        yield minibatch

def get_combined_dirlist(batch_size):
    dirlist = sorted(glob.glob('train_batches'+str(batch_size)+'/*.pickle'))
    for p in sorted(glob.glob('valid_batches'+str(batch_size)+'/*.pickle')):
        dirlist.append(p)
    random.seed(42)
    random.shuffle(dirlist)
    return dirlist

#returns a list of all the batch files contained in the train batch path
def get_train_dirlist(batch_size):
    dirlist = sorted(glob.glob('train_batches'+str(batch_size)+'/*.pickle'))
    return dirlist

#returns a list of all the batch files contained in the valid batch path
def get_valid_dirlist(batch_size):
    dirlist = sorted(glob.glob('valid_batches'+str(batch_size)+'/*.pickle'))
    return dirlist

def get_test_dirlist(batch_size, dir = ''):
    dirlist = sorted(glob.glob(dir + 'test_batches'+str(batch_size)+'/*.pickle'))
    return dirlist

def get_num_pos_samples_and_size(dataframe, batch_size, mode, train_size, valid_size):
    # trainloader = dataIterator(dataframe,batch_size, 0,0.7,0.15)
    indexes = [i for i in range(dataframe.shape[0])]
    random.seed(42)
    random.shuffle(indexes)
    if (mode == 0):
        indexes = indexes[:int(train_size*len(indexes))]
    elif (mode == 1):
        indexes = indexes[int(train_size*len(indexes)):int(train_size*len(indexes))+int(valid_size*len(indexes))]
    else:
        indexes = indexes[int(train_size*len(indexes))+int(valid_size*len(indexes)):]
    index_batches = [[indexes[i] for i in range(n,n+batch_size)] for n in range(0,len(indexes)-batch_size,batch_size)]
    size = len(index_batches) * batch_size
    dataframe = dataframe.to_numpy()
    num_pos_samples = 0
    for batch in index_batches:
        for i in range(batch_size):
            num_pos_samples += dataframe[batch[i],1]
    return num_pos_samples, size

#creates batches by loading the individual images and matching them to the labels; mode = 0 : train, mode = 1 : valid, mode = 2 : test
def dataIterator(df, batch_size,mode, train_size, valid_size, dir = 'train/'):
    #print(df['23e49215068a2bc642fae1cc75cac1e2ea926314'])
    cwd = Path.cwd()
    df = df.to_numpy()
    count = 0
    dirlist = sorted(glob.glob(dir+'*.tif'))
    print(df.shape[0])
    print(batch_size)
    print(len(dirlist))
    indexes = [i for i in range(len(dirlist))]
    random.seed(42)
    random.shuffle(indexes)
    if (mode == 0):
        indexes = indexes[:int(train_size*len(indexes))]
    elif (mode == 1):
        indexes = indexes[int(train_size*len(indexes)):int(train_size*len(indexes))+int(valid_size*len(indexes))]
    else:
        indexes = indexes[int(train_size*len(indexes))+int(valid_size*len(indexes)):]
    index_batches = [[indexes[i] for i in range(n,n+batch_size)] for n in range(0,len(indexes)-batch_size,batch_size)]
    for index_batch in index_batches:
        #print(index_batch)
        batch_imgs = torch.empty((batch_size,3,96,96),dtype=torch.float32)
        batch_labels = torch.empty((batch_size,1),dtype=torch.float32)
        for n in range(batch_size):
            #print("Current File Being Processed is: " + str(i+n))
            infile = dirlist[index_batch[n]]
            #print('file:' + infile + ', label:' + str(df[index_batch[n]]))
            filename = os.path.join(cwd, infile)
            image = cv2.imread(filename)
            img = image.astype(float)/255
            img = img.reshape((3,96,96))
            #print(img)
            #print(img.shape)
            batch_imgs[n]=torch.tensor(img,dtype=torch.float64)
            batch_labels[n]=torch.tensor(float(df[index_batch[n],1]),dtype=torch.float64)
        yield (batch_imgs,batch_labels)

def dataIterator_ids(df, batch_size,mode, train_size, valid_size):
    #print(df['23e49215068a2bc642fae1cc75cac1e2ea926314'])
    cwd = Path.cwd()
    df = df.to_numpy()
    count = 0
    dirlist = sorted(glob.glob('train/*.tif'))
    print(df.shape[0])
    print(batch_size)
    print(len(dirlist))
    indexes = [i for i in range(len(dirlist))]
    random.seed(42)
    random.shuffle(indexes)
    if (mode == 0):
        indexes = indexes[:int(train_size*len(indexes))]
    elif (mode == 1):
        indexes = indexes[int(train_size*len(indexes)):int(train_size*len(indexes))+int(valid_size*len(indexes))]
    else:
        indexes = indexes[int(train_size*len(indexes))+int(valid_size*len(indexes)):]
    index_batches = [[indexes[i] for i in range(n,n+batch_size)] for n in range(0,len(indexes)-batch_size,batch_size)]
    for index_batch in index_batches:
        #print(index_batch)
        batch_imgs = torch.empty((batch_size,3,96,96),dtype=torch.float32)
        batch_labels = torch.empty((batch_size,1),dtype=torch.float32)
        batch_ids = []
        for n in range(batch_size):
            #print("Current File Being Processed is: " + str(i+n))
            infile = dirlist[index_batch[n]]
            #print('file:' + infile + ', label:' + str(df[index_batch[n]]))
            filename = os.path.join(cwd, infile)
            image = cv2.imread(filename)
            img = image.astype(float)/255
            img = img.reshape((3,96,96))
            #print(img)
            #print(img.shape)
            batch_imgs[n]=torch.tensor(img,dtype=torch.float64)
            batch_labels[n]=torch.tensor(float(df[index_batch[n],1]),dtype=torch.float64)
            batch_ids.append(df[index_batch[n],0])
        yield (batch_imgs,batch_labels,batch_ids)
'''
class preloadDataIterator():
    
    def __init__(self, df, batch_size):
        self.cwd = Path.cwd()
        self.df = df.to_numpy()
        self.count = 0
        self.num_splits = 4
        self.batch_size = batch_size
'''        

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
            yield (batch_imgs,batch_labels)'''

#creates an array of all the batches, requires a lot of memory (it is recommended to use iterate batches instead)
def makeTrainDataset(df, size):
    datasetArr = []
    for i, instance in enumerate(dataIterator(df,1)):
        datasetArr.append(instance[0])
        if(i>size-1):
            break
        if(i%10000==0):
            print(i)
    print('finished Loop')
    return datasetArr