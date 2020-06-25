from __future__ import print_function, division
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, FiveCrops)
from torch.autograd import Variable
from selfSupervisedTwoStreamModel import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from selfSupervisedMakeDatasetTwoStream import *
import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./confusion')

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        inputFlow, inputFrame, inputMmap, targets = batch
        inputVariableFlow = Variable(inputFlow.cuda())
        inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
        preds,mmapPrediction = model(inputVariableFlow, inputVariableFrame)
        preds=preds.cpu()
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize):

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

    testBatchSize = 1
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    vid_seq_test = makeDataset(dataset_dir, spatial_transform=spatial_transform, sequence=False, numSeg=1,
                               stackSize=stackSize, fmt='.png', phase='Test', seqLen=seqLen)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = twoStreamAttentionModel(stackSize=5, memSize=512, num_classes=num_classes)
    model.load_state_dict(torch.load(model_state_dict))


    for params in model.parameters():
        params.requires_grad = False

    classes = sorted(os.listdir("/content/drive/My Drive/testingGithub/FPAR_project/GTEA61/processed_frames2/train/S1"))[1:]
    print(classes)
    print(len(classes))

    model.train(False)
    model.cuda()

    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorrTwoStream = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
      test_preds = get_all_preds(model, test_loader)
      labels = vid_seq_test.labels
      predictions = test_preds.argmax(dim=1)
      cm = confusion_matrix(labels, predictions)
      plt.figure(figsize=(25,25))
      plot_confusion_matrix(cm, classes)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/test',
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default='./models/gtea61/best_model_state_dict_twoStream_split2.pth',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize

    main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize)

__main__()