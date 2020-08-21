import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random
import cv2

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
        vid_nameX = self.images[idx].replace('processed_frames2', 'flow_x_processed')
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
            
            #color = None
            if self.color == 'HSV_opticalFlow':
                fl_name = col_name + '/hsv_of_' + str(int(np.floor(i))).zfill(4) + self.fmt
                img = Image.open(fl_name)
                inpSeqCol.append(self.spatial_transform(img.convert('RGB')))
                
            elif self.color == 'flow_surfaceNormals':
                fl_name = col_name + '/flow_surfaceNormal_' + str(int(np.floor(i))).zfill(4) + self.fmt
                img = Image.open(fl_name)
                inpSeqCol.append(self.spatial_transform(img.convert('RGB')))
                
            elif self.color == 'warpedHSV':
                fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt                    
                flow_x = cv2.imread(fl_name,cv2.IMREAD_GRAYSCALE).astype(np.float32)
                flow_y = cv2.imread(fl_name.replace('flow_x','flow_y'),cv2.IMREAD_GRAYSCALE).astype(np.float32)
                
                mag, ang = cv2.cartToPolar(flow_x, flow_y)
                hsv = np.zeros((flow_x.shape[0],flow_x.shape[1],3)).astype(np.uint8)
                hsv[...,1] = 255
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                im_pil = Image.fromarray(rgb)#conversion to a pil image to apply transform
                inpSeqCol.append(self.spatial_transform(im_pil.convert('RGB')))
            elif self.color == 'colorJet':
                ### TO-DO (forse)
                print(self.color,' is not valid')
                exit(-1)
            else:
                print(self.color,' is not valid')
                exit(-1)
            
        
        inpSeqRGB = torch.stack(inpSeqRGB, 0)
        inpSeqCol = torch.stack(inpSeqCol, 0)
        return inpSeqRGB, inpSeqCol, label
