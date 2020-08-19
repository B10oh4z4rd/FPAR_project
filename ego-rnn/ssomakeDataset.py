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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = Normalize(mean=mean, std=std)
spatial_transform2 = Compose([Scale((7,7)), ToTensor()]) 

def listDirectory(path):
  if os.path.isdir(path):
    return os.listdir(path)
  
  return []

def gen_split(root_dir, stackSize):
    DatasetF = []
    Labels = []
    NumFrames = []
    #The root directory should be processed frames/train or test
    for dir_user in sorted(os.listdir(root_dir)):
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(listDirectory(dir1))              
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1,inst,'rgb')
                    inst_dir2 = os.path.join(dir1,inst)
                    numFrames = len(glob.glob1(inst_dir, '*.png'))

                    if numFrames >= stackSize:
                        DatasetF.append(inst_dir2)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
                class_id += 1
    bad_lab = [x for x in Labels if x>=61]
    #print(f'Bad labels {bad_lab}')
    return DatasetF, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesF, self.labels, self.numFrames = gen_split(root_dir, stackSize)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen

    def __len__(self):
        return len(self.imagesF)

    def __getitem__(self, idx):
        vid_nameF = self.imagesF[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()

        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF +  '/' + 'rgb' +'/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)

        #Adding the mmaps to the dataloader
        inpSeqMmaps = []
        try:
          inpSeqMmaps = []
          for i in np.linspace(1,numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF +  '/' + 'mmaps' +'/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqMmaps.append(spatial_transform2(img.convert('L')))
        except:
            inpSeqMmaps = []
            for i in np.linspace(2,numFrame, self.seqLen, endpoint=False):
              fl_name = vid_nameF +  '/' + 'mmaps' +'/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
              img = Image.open(fl_name)
              inpSeqMmaps.append(spatial_transform2(img.convert('L'))) #Grayscale
  
        inpSeqMmaps = torch.stack(inpSeqMmaps,0)

        return inpSeqF, inpSeqMmaps ,label#, vid_nameF#, fl_name