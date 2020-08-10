from __future__ import print_function, division
from regObjectAttentionModelConvLSTM import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
from ssomakeDataset import *
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def main_run(dataset,model_state_dict, dataset_dir, seqLen, memSize,stackSize):

    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
      num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
    sequence = True


    vid_seq_test = makeDataset(dataset_dir, spatial_transform=spatial_transform, stackSize=stackSize, fmt='.png', phase='Test', seqLen=seqLen)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = attentionModel(num_classes=num_classes, mem_size=memSize)
    model.load_state_dict(torch.load(model_state_dict))
    
    for params in model.parameters():
        params.requires_grad = False
    
    model.train(False)
    model.cuda()
    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorr = 0
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        #for j, (inputs, targets) in enumerate(test_loader):
        for inputs, inputMmap, targets in test_loader:
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
            output_label, _ , mmapPrediction = model(inputVariable)
            
            _, predicted = torch.max(output_label.data, 1)
            numCorr += (predicted == targets.cuda()).sum()
            true_labels.append(targets)
            predicted_labels.append(predicted.cpu())
    
    test_accuracy = torch.true_divide(numCorr, test_samples) * 100
    test_accuracy = 'Test Accuracy = {}%'.format(test_accuracy)
    print(test_accuracy)

    

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='./GTEA61', help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default='./models/gtea61/best_model_state_dict_rgb_split2.pth',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')


    args = parser.parse_args()
    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize


    main_run(dataset,model_state_dict, dataset_dir, seqLen, memSize,stackSize)

__main__()