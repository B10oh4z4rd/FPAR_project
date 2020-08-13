import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from objectAttentionModelConvLSTM import *
from noAttentionConvLSTM import *


class crossAttentionDoubleResnet(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, rgbm = None, fcm = None):
        super(crossAttentionDoubleResnet, self).__init__()
        self.num_classes = num_classes
        self.resNet1 = resnetMod.resnet34(True, True)
        self.lstm_cell_x = MyConvLSTMCell(512, mem_size)
        if rgbm is not None:
            model = attentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(rgbm))
            self.resNet1.load_state_dict(model.resNet.state_dict())
            self.lstm_cell_x.load_state_dict(model.lstm_cell.state_dict())
        self.resNet2 = resnetMod.resnet34(True, True)
        if fcm is not None:
            model = noAttentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(fcm))
            self.resNet2.load_state_dict(model.resNet.state_dict())
        self.mem_size = mem_size
        self.weight_softmax = self.resNet1.fc.weight
        
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, inputVariable2):
        state_x = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet1(inputVariable[t])
            logit2, feature_conv2, feature_convNBN2 = self.resNet2(inputVariable2[t])
            bz, nc, h, w = feature_conv2.size()
            feature_conv1 = feature_conv2.view(bz, nc, h*w)
            probs, idxs = logit2.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state_x = self.lstm_cell_x(attentionFeat, state_x)
            
        feats1 = self.avgpool(state_x[1]).view(state_x[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1
