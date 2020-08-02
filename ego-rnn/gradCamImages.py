from selfSupervisedTwoStreamModel import *
from PIL import Image
import grad_cam
from makeDatasetsNames import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import matplotlib.pyplot as plt

import importlib
importlib.reload(grad_cam)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = Normalize(mean=mean, std=std)
spatial_transform = Compose([Scale(256), MultiScaleCornerCrop([1], 224),
                                 ToTensor(), normalize])

def frame_example(image):
  flowModel = "../experiments/gtea61/flow/model_flow_state_dict.pth"
  rgbModel = "modelsFolder/experiments/gtea61/rgb/stage2/model_rgb_state_dict.pth"
  stackSize = 5
  memSize = 512
  num_classes= 61
  seqLen = 7
  model_state_dict = "modelsFolder/selfSupervisedExperiments/gtea61/twoStream/model_twoStream_state_dict.pth"
  trainDatasetDir = "../GTEA61/flow_x_processed/train"
  model = twoStreamAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=stackSize, memSize=memSize,
                                    num_classes=num_classes)
  model.cuda()
  model.load_state_dict(torch.load(model_state_dict))
  model.eval()
  vid_seq_train = makeDataset(trainDatasetDir,spatial_transform=spatial_transform,
                               sequence=False, numSeg=1, stackSize=stackSize, fmt='.png', seqLen=seqLen)
  
  inpFlow, inpFrame, _ , label = vid_seq_train.__getitem__(image)
  heatmap_layer = model.frameModel.resNet.layer4[2].conv2
 # print(inpFlow.size())
 # print(inpFrame.size())
  images = grad_cam.grad_cam(model, inpFlow, inpFrame , heatmap_layer, label)

  for i,image in enumerate(images):
    plt.imshow(image)
    plt.savefig(f'./images/my_grad-cam{i}')