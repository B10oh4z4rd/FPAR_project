import cv2
import torch
import numpy as np

class InfoHolder():

    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()

def generate_heatmap(weighted_activation):
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img * 0.4)
    pil_img = cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB)
    return pil_img

def to_RGB(tensor):
    images = []
    for i in range(7):
      tensorf = tensor[0][i]
      tensorf = (tensorf - tensorf.min())
      tensorf = tensorf/(tensorf.max() + 1e-10)
      print(f'Tensor size in to_RGB {tensorf.size()}')
      image_binary = np.transpose(tensorf.cpu().numpy(), (1, 2, 0))
      image = np.uint8(255 * image_binary)
      images.append(image)
    return images

def grad_cam(model, inpFrame, heatmap_layer, truelabel=None):
    inpFramenormal = inpFrame.unsqueeze(0)
    inpFrame = inpFrame.unsqueeze(0).permute(1, 0, 2, 3, 4).cuda()
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)
    
    output, _ , mmap = model(inpFrame)
    truelabel = truelabel if truelabel else torch.argmax(output)

    print(f'Size of the output {output.size()}')
    output[0][truelabel].backward()

    weights = torch.mean(info.gradient, [0, 2, 3])
    activation = info.activation.squeeze(0)

    weighted_activation = torch.zeros(activation.shape)
    for idx, (weight, activation) in enumerate(zip(weights, activation)):
        weighted_activation[idx] = weight * activation

    heatmap = generate_heatmap(weighted_activation)
    input_images = to_RGB(inpFramenormal)

    cam_visualization = []
    for input_image in input_images:
      cam_visualization.append(superimpose(input_image, heatmap))

    return cam_visualization