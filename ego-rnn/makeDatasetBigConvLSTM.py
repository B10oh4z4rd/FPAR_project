import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

def gen_split(root_dir, stackSize, user, fmt = ".png"):
    fmt = "*" + fmt
    class_id = 0
    
    Dataset = []
    Labels = []
    NumFrames = []
    try:
        dir_user = os.path.join(root_dir, 'processed_frames2', user)
        for target in sorted(os.listdir(dir_user)):
            if target.startswith('.'):
                continue
            target = os.path.join(dir_user, target)
            insts = sorted(os.listdir(target))
            if insts != []:
                for inst in insts:
                    if inst.startswith('.'):
                        continue
                    inst_rgb = os.path.join(target, inst, "rgb")
                    numFrames = len(glob.glob1(inst_rgb, fmt))
                    if numFrames >= stackSize:
                        Dataset.append(target + '/' + inst)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    except:
        print('error')
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1,
                 fmt='.png', users=[], colorization=None):
        self.images = []
        self.labels = []
        self.numFrames = []
        
        for user in users:
            imgs, lbls, nfrms = gen_split(root_dir, seqLen, user, fmt)
            self.images.extend(imgs)
            self.labels.extend(lbls)
            self.numFrames.extend(nfrms)
        
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt
        self.color = colorization
        if train:
            print("Train", end=" ")
        else:
            print("Validation/Test", end=" ")
        print(f'dataset size: {len(self.images)} videos')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        vid_name = self.images[idx] + "/rgb"
        col_name = self.images[idx] + "/" + self.color
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        
        inpSeqRGB = []
        inpSeqCol = []
        self.spatial_transform.randomize_parameters()
        
        for i in np.linspace(1, numFrame, self.seqLen):
            fl_name = vid_name + '/rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqRGB.append(self.spatial_transform(img.convert('RGB')))
            
            color = None
            if self.color == 'HSV_opticalFlow':
                color = '/hsv_of_'
            elif self.color == 'flow_surfaceNormals':
                color = '/flow_surfaceNormal_'
            elif self.color == 'warpedHSV':
                ### TO-DO (forse)
                print(self.color,' is not valid')
                exit(-1)
            elif self.color == 'colorJet':
                ### TO-DO (forse)
                print(self.color,' is not valid')
                exit(-1)
            else:
                print(self.color,' is not valid')
                exit(-1)
            
            fl_name = col_name + color + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqCol.append(self.spatial_transform(img.convert('RGB')))
        
        inpSeqRGB = torch.stack(inpSeqRGB, 0)
        inpSeqCol = torch.stack(inpSeqCol, 0)
        return inpSeqRGB, inpSeqCol, label
