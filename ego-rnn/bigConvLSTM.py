import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from objectAttentionModelConvLSTM import *
from noAttentionConvLSTM import *


class bigConvLSTM(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, rgbm = None, fcm = None):
        super(bigConvLSTM, self).__init__()
        self.num_classes = num_classes
        self.mem_size = mem_size
        
        self.resNetRGB = resnetMod.resnet34(True, True)
        self.resNetCol = resnetMod.resnet34(True, True)
        self.lstm_cell = MyConvLSTMCell(512*2, mem_size)
        
        if rgbm is not None:
            model = attentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(rgbm,
                map_location=torch.device('cpu')))
            self.resNetRGB.load_state_dict(model.resNet.state_dict())
            #self.lstm_cell.load_state_dict(model.lstm_cell)
        if fcm is not None:
            model = noAttentionModel(num_classes, mem_size)
            model.load_state_dict(torch.load(fcm,
                map_location=torch.device('cpu')))
            self.resNetCol.load_state_dict(model.resNet.state_dict())
            #self.lstm_cell_y.load_state_dict(model.lstm_cell)
        
        self.weight_softmax = self.resNetRGB.fc.weight
        
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7) # 0.7
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputRGB, inputCol, device):
        '''
        stateRGB = (Variable(torch.zeros((inputRGB.size(1), self.mem_size, 7, 7)).to(device)),
                 Variable(torch.zeros((inputRGB.size(1), self.mem_size, 7, 7)).to(device)))
        stateCol = (Variable(torch.zeros((inputCol.size(1), self.mem_size, 7, 7)).to(device)),
                 Variable(torch.zeros((inputCol.size(1), self.mem_size, 7, 7)).to(device)))
        state = (torch.cat((stateRGB[0], stateCol[0]), 1),
                torch.cat((stateRGB[1], stateCol[1]), 1))
        '''
        state = (Variable(torch.zeros((inputRGB.size(1), self.mem_size, 7, 7)).to(device)),
                 Variable(torch.zeros((inputCol.size(1), self.mem_size, 7, 7)).to(device)))
        
        
        for t in range(inputRGB.size(0)):
            logit, feature_conv, feature_convNBN = self.resNetRGB(inputRGB[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            
            #probs, idxs = logit.sort(1, True)
            _, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            
            #logit2, feature_conv2, feature_convNBN2 = self.resNetCol(inputCol[t])
            _, _, feature_convNBN2 = self.resNetCol(inputCol[t])
            
            features = torch.cat((attentionFeat, feature_convNBN2), 1)
            state = self.lstm_cell(features, state)
            #stateRGB = self.lstm_cell_x(attentionFeat, stateRGB)
            #stateCol = self.lstm_cell_y(feature_convNBN2, stateCol)
        
        #feats1 = self.avgpool(stateRGB[1]).view(stateRGB[1].size(0), -1)
        #feats2 = self.avgpool(stateCol[1]).view(stateCol[1].size(0), -1)
        #feats_xy = torch.cat((feats1,feats2), 1)
        #feats = self.classifier(feats_xy)
         
        feats_xy = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats_xy)
        return feats, feats_xy
    