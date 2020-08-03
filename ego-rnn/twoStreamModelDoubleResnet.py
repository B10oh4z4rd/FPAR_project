import torch
from flow_resnet import *
from objectAttentionModelConvLSTM import *
from noAttentionConvLSTM import *
import torch.nn as nn


class twoStreamFlowCol(nn.Module):
    def __init__(self, flowModel='', frameModel='', seqLen = 7, memSize=512, num_classes=61):
        super(twoStreamFlowCol, self).__init__()
        self.flowModel = noAttentionModel(num_classes=num_classes, mem_size=memSize)
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))
        self.frameModel = attentionModel(num_classes, memSize)
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))
        self.fc2 = nn.Linear(memSize * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFrame, inputVariableFlow):
        _, flowFeats = self.flowModel(inputVariableFlow)
        _, rgbFeats = self.frameModel(inputVariableFrame)
        twoStreamFeatsDoubleR = torch.cat((flowFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeatsDoubleR)
