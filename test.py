import torch
from torchvision import transforms
from sklearn import metrics
import architecture
import os
import cv2
from pathlib import Path
import glob
from data_loading import iterate_batches
from data_loading import iterate_test_batches
from data_loading import iterate_batches_kfold
from data_loading import get_combined_dirlist
from data_loading import loadDataframe
from data_loading import dataIterator
from data_loading import dataIterator_ids
import numpy as np
from matplotlib import pyplot as plt


def metrics_test_TTA(net,device):
    all_labels = []
    all_preds = []
    all_class_preds = []
    transform = torch.nn.Sequential(
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.3)
    )
    transform.to(device)
    #net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_test_batches(): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            for i in range(4):
                outputs = torch.cat((outputs,net(transform(images))),1)
            outputs = torch.mean(outputs,1,True)
            #print(outputs)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
                all_class_preds.append(int(pred.item()>0.5))
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    # print('accuracy:')
    # print(metrics.accuracy_score(all_labels,all_class_preds))
    # print('recall:')
    # print(metrics.recall_score(all_labels,all_class_preds))
    # print('f1_score:')
    # print(metrics.f1_score(all_labels,all_class_preds))
    accuracy = metrics.accuracy_score(all_labels,all_class_preds)
    recall = metrics.recall_score(all_labels,all_class_preds)
    f1_score = metrics.f1_score(all_labels,all_class_preds)
    
    return ( accuracy,f1_score,auc_score, recall )

def metrics_test(net,device):
    all_labels = []
    all_preds = []
    all_class_preds = []
    #net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_test_batches(): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            #print(outputs)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
                all_class_preds.append(int(pred.item()>0.5))
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    accuracy = metrics.accuracy_score(all_labels,all_class_preds)
    recall = metrics.recall_score(all_labels,all_class_preds)
    f1_score = metrics.f1_score(all_labels,all_class_preds)
    
    return ( accuracy,f1_score,auc_score, recall )


def metrics_and_loss(net, loss_function,device ):
    total_correct = 0 
    total_loss = 0.0 
    total_examples = 0 
    total_positive = 0
    total_true_positive = 0
    n_batches = 0 
    all_labels = []
    all_preds = []
    #net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_batches(128, train=False, shuffle=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            #print(outputs)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
            batch_loss = loss_function(outputs, labels) # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_positive = sum(labels).item()
            total_true_positive = sum((((outputs > 0.5 ) == ( labels > 0.5 ))+labels-1)>0.5).item()
            total_correct += sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item() # number correct in the minibatch
            total_examples += labels.size(0) # the number of labels, which is just the size of the minibatch 
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    accuracy = total_correct / total_examples
    recall = total_true_positive/total_positive
    mean_loss = total_loss / n_batches
    
    return ( accuracy,auc_score, recall, mean_loss )

def metrics_and_loss_TTA(net, loss_function,device ):
    total_correct = 0 
    total_loss = 0.0 
    total_examples = 0 
    total_positive = 0
    total_true_positive = 0
    n_batches = 0 
    all_labels = []
    all_preds = []
    transform = torch.nn.Sequential(
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.3)
    )
    transform.to(device)
    #net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_batches(128, train=False, shuffle=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            for i in range(4):
                outputs = torch.cat((outputs,net(transform(images))),1)
            outputs = torch.mean(outputs,1,True)
            #print(outputs)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
            batch_loss = loss_function(outputs, labels) # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_positive = sum(labels).item()
            total_true_positive = sum((((outputs > 0.5 ) == ( labels > 0.5 ))+labels-1)>0.5).item()
            total_correct += sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item() # number correct in the minibatch
            total_examples += labels.size(0) # the number of labels, which is just the size of the minibatch 
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    accuracy = total_correct / total_examples
    recall = total_true_positive/total_positive
    mean_loss = total_loss / n_batches
    
    return ( accuracy,auc_score, recall, mean_loss )

def metrics_and_loss_kfold( kfolds, currentfold, net, loss_function,device ):
    total_correct = 0 
    total_loss = 0.0 
    total_examples = 0 
    total_positive = 0
    total_true_positive = 0
    n_batches = 0 
    all_labels = []
    all_preds = []
    net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_batches_kfold(kfolds, currentfold, train=False, shuffle=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
            batch_loss = loss_function(outputs, labels) # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_positive = sum(labels).item()
            total_true_positive = sum((((outputs > 0.5 ) == ( labels > 0.5 ))+labels-1)>0.5).item()
            total_correct += sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item() # number correct in the minibatch
            total_examples += labels.size(0) # the number of labels, which is just the size of the minibatch 
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    accuracy = total_correct / total_examples
    recall = total_true_positive/total_positive
    mean_loss = total_loss / n_batches
    
    return ( accuracy,auc_score, recall, mean_loss )

def metrics_and_loss_kfold_TTA( kfolds, currentfold, net, loss_function,device ):
    total_correct = 0 
    total_loss = 0.0 
    total_examples = 0 
    total_positive = 0
    total_true_positive = 0
    n_batches = 0 
    all_labels = []
    all_preds = []
    transform = torch.nn.Sequential(
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.3)
    )
    transform.to(device)
    net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_batches_kfold(kfolds, currentfold, train=False, shuffle=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            for i in range(4):
                outputs = torch.cat((outputs,net(transform(images))),1)
            outputs = torch.mean(outputs,1,True)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
            batch_loss = loss_function(outputs, labels) # this is averaged over the batch
            n_batches += 1
            total_loss += batch_loss.item()
            total_positive = sum(labels).item()
            total_true_positive = sum((((outputs > 0.5 ) == ( labels > 0.5 ))+labels-1)>0.5).item()
            total_correct += sum( (outputs > 0.5 ) == ( labels > 0.5 ) ).item() # number correct in the minibatch
            total_examples += labels.size(0) # the number of labels, which is just the size of the minibatch 
             
    auc_score = metrics.roc_auc_score(all_labels,all_preds)
    accuracy = total_correct / total_examples
    recall = total_true_positive/total_positive
    mean_loss = total_loss / n_batches
    
    return ( accuracy,auc_score, recall, mean_loss )

def get_labels_and_preds(kfolds, currentfold, net,device ):
    all_labels = []
    all_preds = []
    transform = torch.nn.Sequential(
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.3)
    )
    transform.to(device)
    net.eval()
    with torch.no_grad():  # we do not neet to compute the gradients when making predictions on the validation set kfolds = 4, currentfold = 0, batch_size = 128, train=True, shuffle=True
        for data in iterate_batches_kfold(kfolds, currentfold, train=False, shuffle=False): 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            for i in range(4):
                outputs = torch.cat((outputs,net(transform(images))),1)
            outputs = torch.mean(outputs,1,True)
            for label in labels:
                all_labels.append(label.item())
            for pred in outputs:
                all_preds.append(pred.item())
             
    return ( all_labels, all_preds)

def accuracy_at_threshold(labels, preds, threshold):
    y_pred = preds > threshold
    return metrics.accuracy_score(labels,y_pred)

def main():
    model_names = [
        'ResNet_ta_0.910958904109589_va_0.8849957191780822.pickle',
        'ResNet_ta_0.9175442351598173_va_0.8829195205479452.pickle',
        'ResNet_ta_0.9121218607305936_va_0.8868792808219178.pickle',
        'ResNet_ta_0.9134631849315068_va_0.8829623287671233.pickle',
        'ResNet18_ta_0.8717893835616438_va_0.8638056506849315.pickle',
        'ResNet18_ta_0.8759132420091325_va_0.8632063356164383.pickle',
        'ResNet18_ta_0.8895048515981735_va_0.8667166095890411.pickle', #this is the best one, fold 0
        'ResNet18_ta_0.9030679223744292_va_0.8687071917808219.pickle',
        'ResNet18_ta_0.8989226598173516_va_0.8702482876712329.pickle',
        'ResNet18_ta_0.8987157534246575_va_0.8726883561643836.pickle',
        'ResNet_ta_0.919011066084788_va_0.8889834630350194.pickle'
    ]
    use_cuda = torch.cuda.is_available()
    #for debugging it is sometimes useful to set the device to cpu as it usually gives more meaningful error messages
    #device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    loss_function = torch.nn.BCELoss() 
    model = architecture.ResNet18(img_channels=3, num_layers=18, block=architecture.BasicBlock, num_classes=1)
    model.load_state_dict(torch.load('model/ResNet18Holdout_2dropout.pickle'))
    model.to(device)
    print('ResNet18:')
    print('with TTA:')
    print(metrics_test_TTA(model, device))
    print('without TTA:')
    print(metrics_test(model, device))
    model = architecture.ResNet()
    model.load_state_dict(torch.load('model/ResNet_ta_0.9175442351598173_va_0.8829195205479452.pickle'))
    model.to(device)
    print('ResNet10:')
    print('with TTA:')
    print(metrics_test_TTA(model,device))
    print('without TTA:')
    print(metrics_test(model,device))
    
    # print(metrics_and_loss_TTA(model,loss_function,device))
    # all_labels, all_preds = get_labels_and_preds(4,0,model,device)
    # x = np.linspace(0, 1, 1000)
    # y = [accuracy_at_threshold(all_labels,all_preds,threshold) for threshold in x]
    # plt.plot(x, y)
    # plt.show()
    # for name in model_names:
    #     if 'ResNet18' in name:
    #         print("True")
            # model = architecture.ResNet18(img_channels=3, num_layers=18, block=architecture.BasicBlock, num_classes=1)
            # model.load_state_dict(torch.load('model/' + name))
            # model.to(device)
            # for i in range(4):
            #     print(i)
            #     print(metrics_and_loss_kfold(4,i,model,loss_function,device))
        # else:
        #     print("False")
        #     model = architecture.ResNet()
        #     model.load_state_dict(torch.load('model/' + name))
        #     model.to(device)
        #     print(metrics_and_loss(model,loss_function,device))
        #     print(metrics_and_loss_TTA(model,loss_function,device))
    '''model = architecture.ResNet()
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

    '''


if __name__=='__main__':
    main()