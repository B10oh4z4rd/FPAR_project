from __future__ import print_function, division
from bigConvLSTM import bigConvLSTM
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
from makeDatasetBigConvLSTM import makeDataset
from torch.autograd import Variable
import torch
import torch.nn as nn
import datetime
import argparse
import sys
import numpy as np
import os
#import cv2

def main_run(stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize, color, rgbm, fcm):
    #dataset = 'gtea61'
    begin_time = datetime.datetime.now()
    num_classes = 61
    
    if color not in ['HSV_opticalFlow', 'flow_surfaceNormals', 'warpedHSV']:
        print(color,' is not valid')
        exit(-1)
    
    model_folder = os.path.join('./', out_dir, 'BigConvLSTM', color, str(seqLen), 'stage'+str(stage)) # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)
    
    
    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')
    
    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vid_seq_train = makeDataset(train_data_dir, seqLen=seqLen, fmt='.png', users=['S1', 'S3', 'S4'],
                                spatial_transform=spatial_transform, colorization=color)
    #trainInstances = vid_seq_train.__len__()

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    
    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir, seqLen=seqLen, fmt='.png', users=['S2'], train=False,
                                   spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                   colorization=color)
        #valInstances = vid_seq_val.__len__()

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
    
    train_params = []
    if stage == 1:
        model = bigConvLSTM(num_classes=num_classes, mem_size=memSize,
            rgbm=rgbm, fcm=fcm)
        model.train(False)
        
        for params in model.parameters():
            params.requires_grad = False
    else: # stage == 2
        model = bigConvLSTM(num_classes=num_classes, mem_size=memSize,
            rgbm=rgbm, fcm=fcm)
        
        model.load_state_dict(torch.load(stage1_dict))
        model.train(False)
        
        for params in model.parameters():
            params.requires_grad = False
        #
        for params in model.resNetRGB.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetRGB.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetRGB.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetRGB.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetRGB.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        
        for params in model.resNetRGB.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNetRGB.fc.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNetCol.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetCol.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetCol.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetCol.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNetCol.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        
        for params in model.resNetCol.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNetCol.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        model.resNetRGB.layer4[0].conv1.train(True)
        model.resNetRGB.layer4[0].conv2.train(True)
        model.resNetRGB.layer4[1].conv1.train(True)
        model.resNetRGB.layer4[1].conv2.train(True)
        model.resNetRGB.layer4[2].conv1.train(True)
        model.resNetRGB.layer4[2].conv2.train(True)
        model.resNetRGB.fc.train(True)
        
        model.resNetCol.layer4[0].conv1.train(True)
        model.resNetCol.layer4[0].conv2.train(True)
        model.resNetCol.layer4[1].conv1.train(True)
        model.resNetCol.layer4[1].conv2.train(True)
        model.resNetCol.layer4[2].conv1.train(True)
        model.resNetCol.layer4[2].conv2.train(True)
        model.resNetCol.fc.train(True)
    
    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.lstm_cell.train(True)

    model.classifier.train(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0

    dataload_time = datetime.datetime.now()

    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        
        model.lstm_cell.train(True)
        model.classifier.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        
        if stage == 2:
            model.resNetRGB.layer4[0].conv1.train(True)
            model.resNetRGB.layer4[0].conv2.train(True)
            model.resNetRGB.layer4[1].conv1.train(True)
            model.resNetRGB.layer4[1].conv2.train(True)
            model.resNetRGB.layer4[2].conv1.train(True)
            model.resNetRGB.layer4[2].conv2.train(True)
            model.resNetRGB.fc.train(True)
            
            model.resNetCol.layer4[0].conv1.train(True)
            model.resNetCol.layer4[0].conv2.train(True)
            model.resNetCol.layer4[1].conv1.train(True)
            model.resNetCol.layer4[1].conv2.train(True)
            model.resNetCol.layer4[2].conv1.train(True)
            model.resNetCol.layer4[2].conv2.train(True)
            model.resNetCol.fc.train(True)
        
        #for i, (inputs, targets) in enumerate(train_loader):
        for inputsRGB, inputsCol, targets in train_loader:
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            
            inputVariableRGB = Variable(inputsRGB.permute(1, 0, 2, 3, 4).to(device))
            inputVariableCol = Variable(inputsCol.permute(1, 0, 2, 3, 4).to(device))
            labelVariable = Variable(targets.to(device))
            trainSamples += inputsRGB.size(0)
            
            output_label, _ = model(inputVariableRGB, inputVariableCol, device)
            
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(device)).sum()
            epoch_loss += loss.item()
        
        optim_scheduler.step()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = torch.true_divide(numCorrTrain , trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))
        
        if val_data_dir is not None:
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0

            with torch.no_grad():
                #for j, (inputs, targets) in enumerate(val_loader):
                for inputsRGB, inputsCol, targets in val_loader:
                    val_iter += 1
                    val_samples += inputsRGB.size(0)
                    
                    inputVariableRGB = Variable(inputsRGB.permute(1, 0, 2, 3, 4).to(device))
                    inputVariableCol = Variable(inputsCol.permute(1, 0, 2, 3, 4).to(device))
                    labelVariable = Variable(targets.to(device))
                    #labelVariable = Variable(targets.cuda())
                    
                    output_label, _ = model(inputVariableRGB, inputVariableCol, device)
                    
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_epoch += val_loss.item()
                    
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == targets.to(device)).sum()

            val_accuracy = torch.true_divide(numCorr , val_samples) * 100
            avg_val_loss = val_loss_epoch / val_iter
            
            print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
            writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
            writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
            val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
            val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
            
            if val_accuracy > min_accuracy:
                save_path_model = (model_folder + '/model_' + color + '_state_dict.pth')
                torch.save(model.state_dict(), save_path_model)
                min_accuracy = val_accuracy
                print("saved new best model")

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()
    end_time = datetime.datetime.now()
    print('total time elapsed: ',end_time-begin_time)
    print('dataload time: ',dataload_time-begin_time)
    print('training time: ',end_time-dataload_time)
    timers = open((model_folder + '/timings.txt'), 'w')
    timers.write(f"total time elapsed: {end_time-begin_time} \ndataload time: {dataload_time-begin_time} \ntraining time: {end_time-dataload_time}")
    timers.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default='./GTEA61',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default='./GTEA61',
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--color', type=str, default='surfaceNormal', help='which colorized flow use (colorJet,surfaceNormal,warpedHSV,opticalHSV)')
    
    parser.add_argument('--rgbm', type=str, default=None, help='rgbm for stage 1')
    parser.add_argument('--fcm', type=str, default=None, help='fcm for stage 1')
    #rgbm=rgbm, fcm=fcm
    
    args = parser.parse_args()

    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    color = args.color
    rgbm = args.rgbm
    fcm = args.fcm

    main_run(stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize, color, rgbm, fcm)

__main__()
