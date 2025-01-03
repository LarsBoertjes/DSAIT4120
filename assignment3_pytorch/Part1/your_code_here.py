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
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.
    # Do not use for-loops, make use of Pytorch vectorized operations.
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

    
    return torch.rand((1), requires_grad=True) # Placeholder such that the code runs

def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of number of dimensions of the feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix. 
    # Do not use for-loops, make use of Pytorch functionalities.

    return x

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
    # TODO: 3.1 Implement the style loss given the input feature volume and the
    # style feature volume. Note that:
    # - Only the layers given in style_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    # - Implement the gram_matrix function.

    return torch.rand((1), requires_grad=True) # Placeholder such that the code runs

def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss. Normalize by tensor dimension sizes
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.rand((1), requires_grad=True) # Placeholder such that the code runs

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
    # TODO: 5. Calculate the gradient images based on the sobel kernel
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.zeros_like(img, requires_grad=True) # Placeholder such that the code runs

def edge_loss(img1, img2):
    """ Calculates the edge loss based on the mean squared error between the two images.

    # Parameters:
        @img1 (1,2,H,W)
        @img2 (1,2,H,W)
    
    # Returns the edge loss, a torch.tensor of size (1)
    """
    # TODO: 6. Calculate the edge loss
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.rand((1), requires_grad=True) # Placeholder such that the code runs
