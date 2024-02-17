#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
from torchvision.models  import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
import torch
import numpy as np
from scipy import stats,signal
import json
import os


# In[2]:


# Get Data 
data = datasets.Caltech101(root= 'Dataset',download=True)


# In[3]:


# Get color moments of an image
def get_color_moments(grid,cell,image_np): 
    
    color_moments = 3
    # Features array initiated
    features_rgb = np.zeros((grid[0], grid[1], 3, color_moments), dtype=float)
    
    # Iterate over 10 x 10 grid of the image
    for i in range(grid[0]):
        for j in range(grid[1]):

            for channel in range(3):
                # Take one cell from image and flatten it
                cell_rgb = image_np[i*cell[0]:(i+1)*cell[0], j*cell[1]:(j+1)*cell[1],channel].reshape(-1)
                
                f = 1
                # Calculate mean, std, skewness
                mean = np.mean(cell_rgb)
                std = np.std(cell_rgb)
                skew = np.mean((cell_rgb - mean) ** 3)
                
                if skew<0: # If skewness is negative
                    f = -1
                skewness = (np.abs(skew))**(1/3)
                
                # Store in the feature array
                features_rgb[i,j,channel,0] = mean
                features_rgb[i,j,channel,1] = std
                features_rgb[i,j,channel,2] = f*skewness
                
                
    # Flatten the features array
    features_rgb = features_rgb.reshape(-1) 
    return features_rgb


# In[4]:


# Get Histogram of Gradients for an image
def get_hog(grid, cell,image_gray_np):
    
    # number of bins
    num_bins = 9
    
    # Features array initiated
    features_hog = np.zeros((grid[0], grid[1], num_bins), dtype=float)
    
    # Masks
    mask_x = np.array([[-1, 0, 1]])
    mask_y = np.array([[-1], [0], [1]])

    # Gradient magnitude and orientation of the image
    gradient_x = signal.convolve2d(image_gray_np, mask_x,mode='same')
    gradient_y = signal.convolve2d(image_gray_np, mask_y, mode='same')
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.degrees(np.arctan2(gradient_y, gradient_x))
    
    # Make in range (0,360)
    gradient_orientation[gradient_orientation < 0] += 360
    
    # Iterate over the grid to get features
    for i in range(grid[0]):
        for j in range(grid[1]):
            
            cell_orientation = gradient_orientation[i*cell[0]:(i+1)*cell[0], j*cell[1]:(j+1)*cell[1]]
            cell_magnitude = gradient_magnitude[i*cell[0]:(i+1)*cell[0], j*cell[1]:(j+1)*cell[1]]
            
            # Get histogram
            histogram, edges = np.histogram(cell_orientation,bins=num_bins,range=(0, 360),weights=cell_magnitude)
            
            features_hog[i, j, :] = histogram
    
    # Flatten the features array
    features_hog = features_hog.reshape(-1)
    return features_hog


# In[5]:


# Get the Resnet50 output for an image by hooks in intermediate layers: layer3, avgpool, fc
def get_resnet(image):
    
    # Load the model resnet50
    m = resnet50(weights=ResNet50_Weights.DEFAULT) 
    
    # Preprocess the image into 224x224 and to Tensor
    preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    tensor_image = preprocess(image)
    
    outputs = {}
    
    # Create hook function to get output of particular layers in resnet50
    def wrap_hook(name): # wrap the hook function to get layer name as key in the output dictionary
        def hook(module, input, output):
            outputs[name] = output
        return hook
    
    # Layers to be hooked
    layers = {'layer3': m.layer3,'avgpool': m.avgpool,'fc': m.fc}
    
    for name,layer in layers.items():
        layer.register_forward_hook(wrap_hook(name)) # Register the hook
    
    # Pass the image in the pretrained model
    model = m(tensor_image.unsqueeze(0))
    
    # Get the features and reshape to required dimensions
    features_avgpool = outputs['avgpool'].reshape((1024,2)).mean(dim=[1])
    features_layer3 = outputs['layer3'].reshape((1024,14,14)).mean(dim=[1,2])
    features_fc = outputs['fc'].reshape(-1)
    
    return [features_avgpool,features_layer3,features_fc]


# In[7]:


image_id = 0
# Create image dict for all images
image_dict = {'rgb':{},'hog':{},'avgpool':{},'layer3':{},'fc':{}}


# In[8]:


# Traverse the dataset to get features 
for image, label in data:
    
    l = 300
    w = 100
    size = (l,w)
    
    #Process the Image
    image_resized = image.resize(size)
    image_np = np.array(image_resized)
    image_gray = image_resized.convert('L')
    image_gray_np = np.array(image_gray)
    
    # Check for rgb channels
    if len(image_np.shape) < 3:
        print('insufficient channels',image_id)
        image_id += 1
        continue
    
    # Define grid and cell size
    grid = (10, 10)
    cell = (image_np.shape[0]//grid[0],image_np.shape[1]//grid[1])
    
    # Call the rgb, hog, resnet functions
    features_rgb = get_color_moments(grid,cell,image_np)
    features_hog = get_hog(grid,cell,image_gray_np)
    features_resnet = get_resnet(image)
    
    # Get the feature values and convert to list
    image_dict['rgb'][image_id] = features_rgb.tolist()
    image_dict['hog'][image_id] = features_hog.tolist()
    image_dict['avgpool'][image_id] = features_resnet[0].tolist()
    image_dict['layer3'][image_id] = features_resnet[1].tolist()
    image_dict['fc'][image_id] = features_resnet[2].tolist()

    # Increase the id
    image_id += 1
    


# In[10]:


# Define output file name
output_json = 'features.json'

# Store the features into json
with open(output_json, 'w') as json_file: 
    json.dump(image_dict, json_file)


# In[ ]:




