import torch
import torch.optim as optim
from helper_functions import *
# WARNING: Do not import any other libraries or files

def normalize(img, mean, std):
    """ Z-normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    img = (img - mean) / std

    return img 

def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.
    
    # Returns the content loss, a torch.tensor of size (1)
    """
    # TODO: 2. Implement the content loss given the input feature volume and the
    # content feature volume. Note that:
    # - Only the layers given in content_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.

    loss = 0.0

    for layer in content_layers:
        F_l = input_features[layer]
        P_l = content_features[layer]

        loss += torch.nn.functional.mse_loss(F_l, P_l)

    loss /= len(content_layers)
    
    return loss

def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of dimensions of the feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """

    # Get the size of the matrix
    b, c, h, w = x.size()

    # Flatten the matrix
    features = x.view(b, c, -1)

    # Compute Gram matrix
    gram = torch.bmm(features, features.transpose(1, 2))

    # Normalize matrix
    gram /= (h * w)

    return gram

def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.
    
    # Returns the style loss, a torch.tensor of size (1)
    """
    loss = 0.0

    for layer in style_layers:
        # Get the feature maps for the current layers for input and style images
        F_l = input_features[layer]
        A_l = style_features[layer]

        # Compute Gram matrices
        G_F = gram_matrix(F_l)
        G_A = gram_matrix(A_l)

        # Compute MSE between Gram matrices
        loss += torch.nn.functional.mse_loss(G_F, G_A)

    # normalize loss by the number of layers
    loss /= len(style_layers)

    return loss

def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """

    # Difference between neighboring pixels
    diff_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    diff_x = y[:, :, :, 1:] - y[:, :, :, :-1]

    # Squared difference
    tv_loss = torch.sum(diff_y ** 2) + torch.sum(diff_x ** 2)

    # Calculate normalization factor
    batch_size, channels, height, width = y.shape
    normalization_factor = batch_size * channels * (height - 1) * (width - 1)

    tv_loss /= normalization_factor

    return tv_loss

def get_gradient_imgs(img):
    """ Calculates the gradient images based on the sobel kernel.

    # NOTE: 
      1. The gradient image along the x-dimension should be at first position,
         i.e. at out[:,0,:,:], and the gradient image calulated along the y-dimension
         should be at out[:,1,:,:].
      2. Do not use padding for the convolution.
      3. When defining the Sobel kernel, use the finite element approximation of the gradient and approximate the derivative in x-direction according to:
            df / dx  =  f(x+1,y) - f(x-1,y)   (value of left neighbor pixel is subtracted from the value of the right neighbor pixel)
         and the derivative in y-direction according to:
            df / dy  =  f(x,y+1) - f(x,y-1)   (value of bottom neighbor pixel is subtracted from the value of the top neighbor pixel)

    # Parameters:
        @img grayscale image, tensor of size (1,1,H,W)
    
    # Returns the gradient images, concatenated along the second dimension. 
      Size (1,2,H-2,W-2)
    """

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Move kernels to same device as image
    sobel_x = sobel_x.to(img.device)
    sobel_y = sobel_y.to(img.device)

    # Get gradients in x and y directions with convolution
    grad_x = torch.nn.functional.conv2d(img, sobel_x, stride=1, padding=0)
    grad_y = torch.nn.functional.conv2d(img, sobel_y, stride=1, padding=0)

    grad_x = grad_x[:, :, 1:-1, 1:-1]
    grad_y = grad_y[:, :, 1:-1, 1:-1]

    return torch.cat([grad_x, grad_y], dim=1)

def edge_loss(img1, img2):
    """ Calculates the edge loss based on the mean squared error between the two images.

    # Parameters:
        @img1 (1,2,H,W)
        @img2 (1,2,H,W)
    
    # Returns the edge loss, a torch.tensor of size (1)
    """
    edge_loss_value = torch.nn.functional.mse_loss(img1, img2)

    return edge_loss_value