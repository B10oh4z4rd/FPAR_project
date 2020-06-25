import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random
import cv2
import traceback
import copy
import shutil


def generate_HSVOpticalFlow(root_dir):
    new_dir = root_dir+"/HSV_opticalFlow"
    os.makedirs(new_dir)
    prvs = os.path.join(root_dir,'rgb','rgb0001.png')
    print(prvs)
    prvs = cv2.imread(prvs)
    hsv = np.zeros_like(prvs)
    prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
    hsv[...,1] = 255
    for image_index in range(2,1+len(os.listdir(os.path.join(root_dir,'rgb')))):
        #os.path.dirname(os.path.dirname(path))
        frame2 = cv2.imread(root_dir+f'/rgb/rgb{str(int(np.floor(image_index))).zfill(4)}.png')
        next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        save_frame = f'hsv_of_{str(int(np.floor(image_index-1))).zfill(4)}.png'
        save_dir = os.path.join(new_dir,save_frame)
        cv2.imwrite(save_dir,rgb)
        print(f'generated optical flow {image_index-1} / {image_index}')
    

def gen_split(root_dir, stackSize, user, fmt = ".jpg"):
    fmt = "*" + fmt
    class_id = 0
    
    Dataset = []
    Labels = []
    NumFrames = []
    
    try:
        dir_user = os.path.join(root_dir,'processed_frames2', user)
        for target in sorted(os.listdir(dir_user)):
            if target.startswith('.'):
                continue
            target = os.path.join(dir_user, target)
            insts = sorted(os.listdir(target))
            if insts != []:
                for inst in insts:
                    if inst.startswith('.'):
                        continue
                    inst = os.path.join(target, inst)
                    hsv_dir = os.path.join(inst,'HSV_opticalFlow')
                    #if os.path.exists(hsv_dir):
                    #    print('removing: ',hsv_dir)
                    #    shutil.rmtree(hsv_dir)
                        #print(os.path.exists(hsv_dir),os.path.exists(inst))
                    if not (os.path.exists(hsv_dir)):
                        generate_HSVOpticalFlow(inst)
                    numFrames = len(glob.glob1(hsv_dir, fmt))
                    if numFrames >= stackSize:
                        Dataset.append(hsv_dir)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    except:
        print('error')
        traceback.print_exc()
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20, stackSize = 5,train=True, mulSeg=False, numSeg=1,fmt='.png', users=[]):
        
        self.images = []
        self.labels = []
        self.numFrames = []
    
        for us in users:
            imgs, lbls, nfrms = gen_split(root_dir, 5, us, fmt)
            self.images.extend(imgs)
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
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen):
            fl_name = vid_name + '/' + 'hsv_of_' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        
        return inpSeq, label
