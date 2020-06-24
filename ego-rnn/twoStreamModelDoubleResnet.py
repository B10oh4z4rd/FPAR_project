import torch
from flow_resnet import *
from attentionConvLSTMDoubleResnet import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    def __init__(self, flowModel='', frameSNModel='', stackSize=5, memSize=512, num_classes=61):
        super(twoStreamAttentionModel, self).__init__()
        self.flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))
        self.frameSNModel = attentionDoubleResnet(num_classes, memSize)
        if frameSNModel != '':
            self.frameSNModel.load_state_dict(torch.load(frameSNModel))
        self.fc2 = nn.Linear(512 * 3, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFlow, inputVariableFrame, inputVariableSN):
        _, flowFeats = self.flowModel(inputVariableFlow)
        _, rgbSNFeats = self.frameModel(inputVariableFrame,inputVariableSN)
        twoStreamFeatsDoubleR = torch.cat((flowFeats, rgbSNFeats), 1)
        return self.classifier(twoStreamFeatsDoubleR)
