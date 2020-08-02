from __future__ import print_function, division
from objectAttentionModelConvLSTM import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os


def main_run(model_state_dict, dataset_dir, seqLen, memSize, out_dir,color):
    model_folder = os.path.join('./', out_dir, '', str(seqLen))
    #dataset = 'gtea61'
    num_classes = 61
    
    if color == 'opticalHSV':
        from makeDatasetOpticalFlowColorized import makeDataset# as makeDatasetOFC #optical flow
    elif color == 'warpedHSV':
        from makeDatasetFlowHSVMapped import makeDataset# as makeDatasetWFC #warped flow
    elif color == 'surfaceNormal':
        from makeDatasetFlowSN import makeDataset# as makeDatasetFlowSN #warped flow surface normal
    elif color == 'colorJet':
        from makeDatasetFlowColorized import makeDataset#
    else:
        print(color,' is not valid')
        exit(-1)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
    
    vid_seq_test = makeDataset(dataset_dir, seqLen=seqLen, fmt='.png', train=False,
                               spatial_transform=spatial_transform, users=['S2'])
    
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
        for inputs, targets in test_loader:
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
            output_label, _ = model(inputVariable)
            
            _, predicted = torch.max(output_label.data, 1)
            numCorr += (predicted == targets.cuda()).sum()
            true_labels.append(targets)
            predicted_labels.append(predicted.cpu())
    
    test_accuracy = torch.true_divide(numCorr, test_samples) * 100
    test_accuracy = 'Test Accuracy = {}%'.format(test_accuracy)
    print(test_accuracy)
    fil = open(model_folder + "/test_log_acc.txt", "w")
    fil.write(test_accuracy)
    fil.close()
    
    cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    
    ticks = np.linspace(0, 60, num=61)
    plt.figure(1, figsize=(12, 12), dpi=100.0)
    plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='binary')
    plt.colorbar()
    plt.xticks(ticks, fontsize=6)
    plt.yticks(ticks, fontsize=6)
    plt.grid(True)
    plt.clim(0, 1)
    xy = np.arange(start=0, stop=61)
    plt.plot(xy,xy)
    plt.savefig(model_folder + '/cnf_matrix_normalized.png', bbox_inches='tight')
    plt.show()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', type=str, default='./GTEA61', help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default='./models/gtea61/best_model_state_dict_rgb_split2.pth',
                        help='Model path')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results and images')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--color', type=str, default='surfaceNormal', help='which colorized flow use (colorJet,surfaceNormal,warpedHSV,opticalHSV)')

    args = parser.parse_args()

    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    memSize = args.memSize
    outDir = args.outDir
    color = args.color

    main_run(model_state_dict, dataset_dir, seqLen, memSize, outDir,color)

__main__()
