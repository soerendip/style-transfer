#!/usr/bin/env python

import os, sys

from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

from io import BytesIO
from PIL import Image
import argparse


def load_image(img_path, max_size=1024, shape=None):
    '''Load in and transform an image, making sure the image
       is <= 1024 pixels in the x-y dims.'''
    image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## completing the mapping layer names of PyTorch's VGGNet to names from the paper
    ## we need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    ## get the batch_size, depth, height, and width of the Tensor
    batch_size, c, h, w = tensor.size()
    
    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.reshape(c, h * w)
    
    ## calculate the gram matrix
    gram = torch.mm(tensor, torch.transpose(tensor, 0, 1))
    
    return gram

  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the output")
    parser.add_argument("--input", help="Input picture")
    parser.add_argument("--style", help="Style picture")
    parser.add_argument("--device", help="CUDA device", type=int, default=0)
    parser.add_argument("--style-weight", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--weights", default="0.9,0.6,0.3,0.15,0.0")

    args = parser.parse_args()
    
    weights = [float(i) for i in args.weights.split(',')]
    assert len(weights) == 5, weights
    
    print(args)
    w_style = args.style_weight
    torch.cuda.set_device(args.device)
    
    # Project name
    NAME = args.name
    # Input picture
    raw_input_image_path = args.input
    # Style picture
    raw_style_image_path = args.style
    
    #setting custom path for model download
    os.environ['TORCH_HOME'] = "models/"
    vgg = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
          
    # Output
    OUTPUT_DIR = os.path.join('output', NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=False)
    
    IMAGE_SIZE = 1024
    IMAGE_WIDTH = IMAGE_SIZE
    IMAGE_HEIGHT = IMAGE_SIZE
    
    print('Writing output to:', OUTPUT_DIR)
    print('Input picture:', raw_input_image_path)
    print('Style picture:', raw_style_image_path)
    print('CUDA device:', torch.cuda.current_device())
    print('Weights', weights)
    print('Style weight', w_style)
    
    # Prepared input and style
    input_image_path = os.path.join(OUTPUT_DIR, 'input.png')
    style_image_path = os.path.join(OUTPUT_DIR, 'style.png')
    
    #Input visualization 
    input_image = Image.open(raw_input_image_path)
    input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    input_image.save(input_image_path)
    
    # Style visualization 
    style_image = Image.open(raw_style_image_path)
    style_image = style_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    style_image.save(style_image_path)
    
    # load in content and style image
    content = load_image(input_image_path).to(device)

    # Resize style to match content, makes code easier
    style = load_image(raw_style_image_path, shape=content.shape[-2:]).to(device)

    # get content and style features only once before forming the target image
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image and prep it for change
    # it is a good idea to start of with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)

    # weights for each style layer 
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    
    w = weights
    style_weights = {'conv1_1': w[0],
                     'conv2_1': w[1],
                     'conv3_1': w[2],
                     'conv4_1': w[3],
                     'conv5_1': w[4]}

    # you may choose to leave these as is
    content_weight = 1  # alpha
    style_weight = w_style  # beta

    # for displaying the target image, intermittently
    show_every = 200

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = args.steps  # decide how many iterations to update your image


    for ii in range(1, steps+1):

        ## get the features from your target image    
        ## Then calculate the content loss
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            ## Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)

            ## get the "style" style representation
            style_gram = style_grams[layer]

            ## Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)


        ## calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            print('Step: ', ii)
            plt.imshow(im_convert(target))
            plt.xticks([])
            plt.yticks([])
            #plt.show()
            img = im_convert(target)
            mpl.image.imsave(os.path.join(OUTPUT_DIR, f'{NAME}-step-{ii:05d}.png'), img)
            mpl.image.imsave(f'/var/www/html/dl/styletransfer-{torch.cuda.current_device()}.png', img)

    mpl.image.imsave(os.path.join(OUTPUT_DIR, f'{NAME}.png'), img)
            
