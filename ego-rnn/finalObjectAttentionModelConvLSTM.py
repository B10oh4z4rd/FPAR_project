import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        #Secondary task branch for flowx
        self.xPredictor = nn.Sequential()
        self.xPredictor.add_module('mmap_relu',nn.ReLU(True))
        self.xPredictor.add_module('convolution', nn.Conv2d(512, 100, kernel_size=1))
        self.xPredictor.add_module('flatten',Flatten())
        self.xPredictor.add_module('fc_2',nn.Linear(100*7*7,7*7))

        self.yPredictor = nn.Sequential()
        self.yPredictor.add_module('mmap_relu',nn.ReLU(True))
        self.yPredictor.add_module('convolution', nn.Conv2d(512, 100, kernel_size=1))
        self.yPredictor.add_module('flatten',Flatten())
        self.yPredictor.add_module('fc_2',nn.Linear(100*7*7,7*7))



    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

            #Prediction of the x flow
            feature_conv2 = feature_conv.clone()
            feature_conv3 = feature_conv.clone()

            if t==0:
              flowXprediction = self.xPredictor(feature_conv2)
              flowYprediction = self.yPredictor(feature_conv3)
            else:
              predictionX = self.xPredictor(feature_conv2)
              predictionY = self.yPredictor(feature_conv3)

              flowXprediction = torch.cat([flowXprediction,predictionX],dim=0)
              flowYprediction = torch.cat([flowYprediction,predictionY],dim=0)

            state = self.lstm_cell(attentionFeat, state)
        
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, flowXprediction, flowYprediction #Makes the list a stack