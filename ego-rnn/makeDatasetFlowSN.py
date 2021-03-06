import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize, fmt,users):
    Dataset = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'processed_frames2')
    for dir_user in sorted(os.listdir(root_dir)):
        if dir_user not in users:
            continue
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            if not os.path.isdir(dir1):
              continue
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    inst_dir = os.path.join(inst_dir,'flow_surfaceNormals')
                    termination = f'*{fmt}'
                    numFrames = len(glob.glob1(inst_dir, termination))
                    if numFrames >= stackSize:
                        Dataset.append(inst_dir)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    print(f'Dataset size: {len(Dataset)} images')
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir ,spatial_transform=None, seqLen=20, stackSize = 5,
                 train=True, mulSeg=False, numSeg=1, fmt='.png', users = []):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5, fmt, users)
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'flow_surfaceNormal_' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
    
    
