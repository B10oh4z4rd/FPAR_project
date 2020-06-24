import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class attentionDoubleResnet(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(noAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet1 = resnetMod.resnet34(True, True)
        self.resNet2 = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell_x = MyConvLSTMCell(512, mem_size)
        self.lstm_cell_y = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(2 * mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, inputVariable2):
        state_x = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        state_y = (Variable(torch.zeros((inputVariable2.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable2.size(1), self.mem_size, 7, 7)).cuda()))
        
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet1(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state_x = self.lstm_cell_x(attentionFeat, state_x)
            
            logit2, feature_conv2, feature_convNBN2 = self.resNet2(inputVariable2[t])
            bz2, nc2, h2, w2 = feature_conv2.size()
            feature_conv2 = feature_conv2.view(bz, nc, h*w)
            probs2, idxs2 = logit2.sort(1, True)
            class_idx2 = idxs2[:, 0]
            cam2 = torch.bmm(self.weight_softmax[class_idx2].unsqueeze(1), feature_conv2)
            attentionMAP2 = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP2 = attentionMAP2.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat2 = feature_convNBN2 * attentionMAP2.expand_as(feature_conv)
            state_y = self.lstm_cell_y(attentionFeat, state_y)
            
        feats1 = self.avgpool(state[1]).view(state_x[1].size(0), -1)
        feats2 = self.avgpool(state[1]).view(state_y[1].size(0), -1)
        feats_xy = torch.cat((feats1,feats2), 1)
        feats = self.classifier(feats_xy)
        return feats, feats_xy
