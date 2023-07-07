import architecture
import torch
import torch.nn as nn
import torch.nn.functional as F  # this includes tensor functions that we can use in backwards pass
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib
matplotlib.use('WebAgg',force=True)
import matplotlib.pyplot as plt
import pickle
from data_loading import dataIterator
from data_loading import loadDataframe
from data_loading import makeTrainDataset
from data_loading import iterate_batches
from data_loading import get_dirlist_batches
import pandas as pd
from pathlib import Path
import random
from console_parameter_management import get_params
from tqdm import tqdm


def metrics_and_loss( net, loss_function,split,dirlist,device ):
    total_correct = 0 
    total_loss = 0.0 
    total_examples = 0 
    total_positive = 0
    total_true_positive = 0
    n_batches = 0 
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set
        for data in iterate_batches(dirlist,split,train=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            batch_loss = loss_function(outputs, labels) # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_positive = sum(labels).item()
            total_true_positive = sum((((outputs > 0.5 ) == ( labels > 0.5 ))+labels-1)>0.5).item()
            total_correct += sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item() # number correct in the minibatch
            total_examples += labels.size(0) # the number of labels, which is just the size of the minibatch 
             
    
    accuracy = total_correct / total_examples
    recall = total_true_positive/total_positive
    mean_loss = total_loss / n_batches
    
    return ( accuracy, recall, mean_loss )

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def main():
    model_name, save_name, pretrained, pretrained_path, early_stopping = get_params()
    use_cuda = torch.cuda.is_available()
    #for debugging it is sometimes useful to set the device to cpu as it usually gives more meaningful error messages
    #device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training on device:"+str(device))

    dataframe = loadDataframe()
    loss_function = nn.BCELoss() 
    n_epochs = 10
    batch_size=128
    weight_decay=0.0
    split = int((dataframe.shape[0]/batch_size)*0.8)
        

        
    
    match model_name:
        case 'LeNet_kaiming_normal':
            thenet = architecture.LeNet_kaiming_normal()
        case 'LeNet_kn':
            thenet = architecture.LeNet_kaiming_normal()
        case 'LeNet':
            thenet = architecture.LeNet()
        case 'NiN':
            thenet = architecture.NiN()
        case 'ResNet':
            thenet = architecture.ResNet()
        case 'resnet':
            thenet = architecture.ResNet()
        case _:
            model_name = 'LeNet_kn'
            thenet = architecture.LeNet_kaiming_normal()
    if(pretrained):
        thenet.load_state_dict(torch.load(pretrained_path))

    thenet.to(device)
    optimizer1 = optim.Adam( thenet.parameters(), weight_decay=weight_decay )

    val_rec = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    dirlist = get_dirlist_batches(batch_size)
    random.seed(42)
    random.shuffle(dirlist)
    early_stopper = EarlyStopper(patience=3, min_delta=0)

    for epoch in range(n_epochs): # number of times to loop over the dataset
        
        total_loss = 0 
        total_correct = 0 
        total_examples = 0 
        n_mini_batches = 0
        #trainloader = dataIterator(dataframe,batch_size)
        
        trainloader = iterate_batches(dirlist,split)
        for i, mini_batch in tqdm(enumerate(trainloader, 0), unit="batch", total=split):
        #for i, mini_batch in enumerate( trainloader, 0):
            images, labels = mini_batch
            #print(i)
            images, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            # all the parameters that are being updated are in the optimizer, 
            # so if we zero the gradients of all the tensors in the optimizer, 
            # that is the safest way to zero all the gradients
            optimizer1.zero_grad()
            #print(images)
            outputs = thenet(images) # this is the forward pass

            loss = loss_function ( outputs, labels )

            loss.backward() # does the backward pass and computes all gradients

            optimizer1.step() # does one optimisation step

            n_mini_batches += 1 # keep track of number of minibatches, and collect the loss for each minibatch
            total_loss += loss.item() # remember that the loss is a zero-order tensor
            # so that to extract its value, we use .item(), as we cannot index as there are no dimensions

            # keep track of number of examples, and collect number correct in each minibatch
            total_correct += sum( ( outputs > 0.5 ) == ( labels > 0.5 ) ).item()
            total_examples += len( labels )

        # calculate statistics for each epoch and print them. 
        # You can alter this code to accumulate these statistics into lists/vectors and plot them
        epoch_training_accuracy = total_correct / total_examples
        epoch_training_loss = total_loss / n_mini_batches

        epoch_val_accuracy, epoch_val_recall, epoch_val_loss = metrics_and_loss( thenet, loss_function, split,dirlist,device)

        print('Epoch %d loss: %.3f acc: %.3f val_loss: %.3f val_acc: %.3f val_rec: %.3f'
                %(epoch+1, epoch_training_loss, epoch_training_accuracy, epoch_val_loss, epoch_val_accuracy, epoch_val_recall ))
        
        train_loss.append( epoch_training_loss )
        train_acc.append( epoch_training_accuracy )
        val_loss.append( epoch_val_loss )
        val_acc.append( epoch_val_accuracy )
        val_rec.append( epoch_val_recall)
        if early_stopper.early_stop(epoch_val_loss):             
            break

    history = { 'train_loss': train_loss, 
                'train_acc': train_acc, 
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_rec': val_rec }
    cwd = Path.cwd()
    if save_name == 'default':
        torch.save(thenet.state_dict(), f"{cwd}/model/{model_name}_ta_{train_acc[-1]}_va_{val_acc[-1]}.pickle")
    else:
        torch.save(thenet.state_dict(), f"{cwd}/model/{save_name}.pickle")
    plt.plot( history['train_acc'], label='train_acc')
    plt.plot( history['val_acc'], label='val_acc')
    plt.plot( history['val_rec'], label='val_rec')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()