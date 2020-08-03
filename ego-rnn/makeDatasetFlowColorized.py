import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2

def gen_split(root_dir, stackSize, fmt = '.png', dataset = []):
    DatasetX = []
    DatasetY = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'flow_x_processed')
    print(dataset)
    for dir_user in sorted(os.listdir(root_dir)):
        if dir_user not in dataset:
            continue
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    numFrames = len(glob.glob1(inst_dir, fmt))
                    if numFrames >= stackSize:
                        DatasetX.append(inst_dir)
                        DatasetY.append(inst_dir.replace('flow_x', 'flow_y'))
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return DatasetX, DatasetY, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5, seqLen = 16,
                 train=True, numSeg = 1, fmt='.png', phase='train', users = ['S1','S2','S3','S4']):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imagesX, self.imagesY, self.labels, self.numFrames = gen_split(root_dir, seqLen, '*' + fmt, users)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = sequence
        self.fmt = fmt
        self.phase = phase
        self.info_ = {"definition" : "creates a flow dataset",
                      "input parameters" : {
                          "root_dir" : "(str): folder where it searches for the images",
                          "spatial_transform" : "(callable, optional) [default: None]: Opt. transform to be applied on a sample",
                          "sequence" : "(bool): check param numSeg and method __getitem__",
                          "stackSize" : "(int) [default: 5] : defines the num of frames",
                          "train" : "(bool) [default: True]: if it should be a training or test dataset",
                          "numSeg" : "(int) [default: 1]: if sequence is true it defines the number of frames",
                          "fmt" : "(str) [default: '-png']: format of the image to search",
                          "phase" : "defines training or validation phase for defining a random or defined startframe"
                      },
                          
                      "methods" : {
                        "__info__" : "returns a dict containing information about class and dataset",
                        "__len__" : "length of the document",
                        "__getitem__" : "given an index returns a list of list of frames, and the associated label"
                      },
                      "attributes" : {
                        "images_x" : type(self.imagesX),
                        "images_y" : type(self.imagesY),
                        "labels" : type(self.labels), 
                        "numFrames" : type(self.numFrames),
                        "spatial_transform" : self.spatial_transform,
                        "train" : self.train,
                        "numSeg" : self.numSeg,
                        "sequence" : self.sequence,
                        "stackSize" : self.stackSize,
                        "fmt" : self.fmt,
                        "phase" : self.phase
                      }
         }

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()
        if self.sequence is True:
            if numFrame <= self.stackSize:
                frameStart = np.ones(self.numSeg)
            else:
                frameStart = np.linspace(1, numFrame - self.stackSize + 1, self.numSeg, endpoint=False)
            for startFrame in frameStart:
                inpSeq = []
                for k in range(self.stackSize):
                    i = k + int(startFrame)
                    f1_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt
                    #img = Image.open(fl_name)
                    #img = self.spatial_transform(img.convert('L'), inv=True, flow=True)
                    #open_cv_image = numpy.array(pil_image) 
                    # Convert RGB to BGR 
                    #open_cv_image = open_cv_image[:, :, ::-1].copy() 
                    im_gray_x = cv2.imread(f1_name, cv2.IMREAD_GRAYSCALE)
                    im_color_x = cv2.applyColorMap(im_gray_x, cv2.COLORMAP_JET)   
                    # fl_names.append(fl_name)
                    f1_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + self.fmt
                    im_gray_y = cv2.imread(f1_name, cv2.IMREAD_GRAYSCALE)
                    im_color_y = cv2.applyColorMap(im_gray_y, cv2.COLORMAP_JET)   
                    im_max = np.maximum(im_color_x,im_color_y) #fusion of the 2 images to obtain a single frame
                    im_pil = Image.fromarray(im_max)#conversion to a pil image to apply transform
                    inpSeq.append(self.spatial_transform(im_pil))
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
            return inpSeqSegs, label
        else:
            if numFrame <= self.stackSize:
                startFrame = 1
            else:
                if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
                else:
                    startFrame = np.ceil((numFrame - self.stackSize)/2)
            inpSeq = []
            for k in range(self.stackSize):
                i = k + int(startFrame)
                f1_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt
                im_gray_x = cv2.imread(f1_name, cv2.IMREAD_GRAYSCALE)
                im_color_x = cv2.applyColorMap(im_gray_x, cv2.COLORMAP_JET)   
                # fl_names.append(fl_name)
                f1_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + self.fmt
                im_gray_y = cv2.imread(f1_name, cv2.IMREAD_GRAYSCALE)
                im_color_y = cv2.applyColorMap(im_gray_y, cv2.COLORMAP_JET)   
                im_max = np.maximum(im_color_x,im_color_y)
                im_pil = Image.fromarray(im_max)#conversion to a pil image to apply transform
                inpSeq.append(self.spatial_transform(im_pil))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
            return inpSeqSegs, label#, fl_name

    def __info__(self):
      return self.info_
