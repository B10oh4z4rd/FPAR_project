import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
finals =  Compose([ToTensor(), normalize])

def listDirectory(path):
  if os.path.isdir(path):
    return os.listdir(path)
  
  return []

def gen_split(root_dir, stackSize):
    DatasetX = []
    DatasetY = []
    DatasetF = []
    Labels = []
    NumFrames = []
    #The root directory should be flow_x_processed/train or test
    for dir_user in sorted(os.listdir(root_dir)):
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(listDirectory(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    numFrames = len(glob.glob1(inst_dir, '*.png'))
                    if numFrames >= stackSize:
                        DatasetX.append(inst_dir)
                        DatasetY.append(inst_dir.replace('flow_x_processed', 'flow_y_processed'))
                        DatasetF.append(inst_dir.replace('flow_x_processed', 'processed_frames2'))
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return DatasetX, DatasetY, DatasetF, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform2 ,spatial_transform=None,sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.imagesF, self.labels, self.numFrames = gen_split(
            root_dir, stackSize)
        self.spatial_transform2 = spatial_transform2
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        vid_nameF = self.imagesF[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()


        inpSeqX = []
        inpSeqY = []
        for k in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            i = k
            fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
            img = Image.open(fl_name)
            img = self.spatial_transform(img.convert('L'))
            inpSeqX.append(self.spatial_transform2(img))

            fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
            img = Image.open(fl_name)
            img = self.spatial_transform(img.convert('L'))
            inpSeqY.append(self.spatial_transform2(img))

        inpSeqSegsX = torch.stack(inpSeqX, 0)#.squeeze(1)
        inpSeqSegsY = torch.stack(inpSeqY, 0)#.squeeze(1)
            
        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF +  '/' + 'rgb' +'/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            img = self.spatial_transform(img.convert('RGB'))
            inpSeqF.append(finals(img))

        inpSeqF = torch.stack(inpSeqF, 0)
        return inpSeqSegsX, inpSeqSegsY , inpSeqF, label#, vid_nameF#, fl_name