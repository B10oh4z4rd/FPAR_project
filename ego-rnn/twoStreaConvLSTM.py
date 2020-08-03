import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from objectAttentionModelConvLSTM import *
from noAttentionConvLSTM import *

class commonConvLSTM(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, rgbm = None, fcm = None):
        super(attentionDoubleResnet, self).__init__()
        self.num_classes = num_classes
        self.resNet1 = resnetMod.resnet34(True, True)
        if rgbm is not None:
            model = attentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(rgbm))
            self.resNet1.load_state_dict(model.resNet.state_dict())
        self.resNet2 = resnetMod.resnet34(True, True)
        if fcm is not None:
            model = noAttentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(rgbm))
            self.resNet2.load_state_dict(model.resNet.state_dict())
        self.mem_size = mem_size
        self.weight_softmax = self.resNet1.fc.weight
        self.lstm_cell = MyConvLSTMCell(512*2, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(2 * mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, inputVariable2):
        state = (Variable(torch.zeros((inputVariable.size(1) + inputVariable2.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1) + inputVariable2.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet1(inputVariable[t])
            logit2, feature_conv2, feature_convNBN2 = self.resNet2(inputVariable2[t])
            feat_conv = torch.cat((feature_conv,feature_conv2),1)
            state = self.lstm_cell(feature_conv, state)
            
        feats = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats)
        return feats, feats_xy
