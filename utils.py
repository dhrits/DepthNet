import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# Some utilities for drawing

def get_visualized_batch(batch, num_rows=4, means=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    mean = torch.tensor(means).to(device).view(1, 3, 1, 1)
    std = torch.tensor(std).to(device).view(1, 3, 1, 1)
    images = batch[0]
    depths = batch[1]
    images = images * std + mean
    images = images.cpu()
    depths = depths.cpu()
    image_grid = torchvision.utils.make_grid(images, nrow=num_rows)
    image_grid = image_grid.numpy().transpose(1, 2, 0)
    depth_grid = torchvision.utils.make_grid(depths, nrow=num_rows, normalize=True, scale_each=True)
    depth_grid = depth_grid.numpy().transpose(1, 2, 0)
    return image_grid, depth_grid

def get_visualized_results(preds, num_rows=4):
    depth_grid =  torchvision.utils.make_grid(preds.cpu(), nrow=num_rows, normalize=True, scale_each=True)
    depth_grid = depth_grid.numpy().transpose(1, 2, 0)
    return depth_grid

def draw_results(images, depths, preds, device='cpu', means=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_samples=4):
    mean = torch.tensor(means).to(device).view(1, 3, 1, 1)
    std = torch.tensor(std).to(device).view(1, 3, 1, 1)
    images = images * std + mean
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    depths = depths.cpu().numpy().transpose(0, 2, 3, 1)
    preds = preds.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(num_samples, 3)
    for i in range(num_samples):
        axs[i][0].axis('off')
        axs[i][0].imshow(images[i])
    
        axs[i][1].axis('off')
        axs[i][1].imshow(depths[i])
        
        axs[i][2].axis('off')
        axs[i][2].imshow(preds[i])
        
    plt.show()
