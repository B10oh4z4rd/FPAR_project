import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class noAttentionDoubleResnet(nn.Module):
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
            logit2, feature_conv2, feature_convNBN2 = self.resNet2(inputVariable2[t])
            state_x = self.lstm_cell_x(feature_convNBN, state_x)
            state_y = self.lstm_cell_y(feature_convNBN2, state_y)
        feats1 = self.avgpool(state[1]).view(state_x[1].size(0), -1)
        feats2 = self.avgpool(state[1]).view(state_y[1].size(0), -1)
        feats_xy = np.concatenate(feats1,feats2)
        feats = self.classifier(feats_xy)
        return feats, feats_xy
