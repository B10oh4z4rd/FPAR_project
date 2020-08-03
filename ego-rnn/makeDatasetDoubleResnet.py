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
    DatasetSN = []
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
                    inst_rgb  = os.path.join(target, inst, "rgb")
                    inst_SN  = os.path.join(target, inst, "flow_surfaceNormals")
                    numFrames = len(glob.glob1(inst_rgb, fmt))
                    if numFrames >= stackSize:
                        Dataset.append(inst_rgb)
                        DatasetSN.append(inst_SN)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    except:
        print('error')
    return Dataset, DatasetSN, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1,
                 fmt='.png', users=[]):
        self.images = []
        self.surNorms = []
        self.labels = []
        self.numFrames = []
        
        for user in users:
            imgs, surfs, lbls, nfrms = gen_split(root_dir, seqLen, user, fmt)
            self.images.extend(imgs)
            self.surNorms.extend(surfs)
            self.labels.extend(lbls)
            self.numFrames.extend(nfrms)
        
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt
        if train:
            print("Train", end=" ")
        else:
            print("Validation/Test", end=" ")
        print(f'dataset size: {len(self.images)} videos')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        vid_name = self.images[idx]
        SN_name = self.surNorms[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        
        inpSeq = []
        inpSeqSN = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
            flsn_name = SN_name + '/' + 'flow_surfaceNormal_' + str(int(np.floor(i))).zfill(4) + self.fmt
            imgsn = Image.open(flsn_name)
            inpSeqSN.append(self.spatial_transform(imgsn.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        inpSeqSN = torch.stack(inpSeqSN, 0)
        return inpSeq, inpSeqSN, label
