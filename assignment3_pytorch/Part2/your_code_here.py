import torch
import torch.nn.functional as Func
from helper_functions import *
# WARNING: Do not import any other libraries or files

def get_neighbourhood(p, radius):
    """ Returns the neighbourhood grid of points around p.

    NOTE: The neighbourhood should contain (radius * 2 + 1)**2 points, including p.
    For example, for p = [3.5, 3.5] and radius = 1, this function should return the following values:
        (2.5, 2.5) (2.5, 3.5) (2.5, 4.5) 
        (3.5, 2.5) (3.5, 3.5) (3.5, 4.5) 
        (4.5, 2.5) (4.5, 3.5) (4.5, 4.5)
    The order of the points does not matter but has to be consistent inbetween calls.

    # Parameters:
        @p: torch.tensor size [2], the current handle point p
        @radius: int, the radius of the grid neighbourhood to return

    # Returns: torch.tensor size [(radius * 2 + 1)**2, 2], the neighbourhood of points around p, including p
    """
    # TODO: 1. Get Neighbourhood
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.zeros((radius * 2 + 1)**2, device=p.device) # Placeholder such that the code runs

def sample_p_from_feature_map(q_N, F):
    """ Samples the feature map F at the points q_N.

    NOTE: As the points in q_N are floats, we can not directly sample the points from the feature map via indexing.
    Bilinear interpolation is needed, PyTorch has a function for this: F.grid_sample.
    If you struggle with this function, try to set up a minimal working example, where you manually control q_N and F.

    # Parameters:
        @q_N: torch.tensor size [N, 2], the points to sample from the feature map
        @F: torch.tensor size [1, C, H, W], the feature map of the current image

    # Returns: torch.tensor size [N, C], the sampled features at q_N
    """
    assert F.shape[-1] == F.shape[-2]

    # TODO: 2. Sample features from neighbourhood
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.zeros((q_N.shape[0], F.shape[1]), device=q_N.device, requires_grad=True) # Placeholder such that the code runs

def nearest_neighbour_search(f_p, F_q_N, q_N):
    """ Does a nearest neighbourhood search in feature space to find the new handle point position.

    NOTE: The nearest neighbour should be determined by the L1 distance.

    # Parameters:
        @f_p: torch.tensor size [1, C], the feature vector of the handle point p
        @F: torch.tensor size [1, C, H, W], the feature map of the current image
        @q_N: torch.tensor size [N, 2], corresponding points to F_q_N in the image space

    # Returns: torch.tensor size [2], the new handle point p 
    """
    # TODO: 3. Neighbourhood search
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.rand((2)) # Placeholder such that the code runs

def get_mask_loss(F_1, F_2, mask):
    """ Returns the mask loss.

    # Parameters:
        @F_1: torch.tensor size [1, C, H, W], the feature map of the first image
        @F_2: torch.tensor size [1, C, H, W], the feature map of the second image
        @mask: torch.tensor size [H, W], the segmentation mask. 
            NOTE: 1 encodes what areas should move and 0 what areas should stay fixed.
        @lambda_mask: float, the weight of the mask loss
    
    # Returns: torch.tensor of size [1], the mask loss
    """
    # TODO: 4. Mask loss
    # Do not use for-loops, make use of Pytorch vectorized operations.

    return torch.rand((2), device=F_1.device, requires_grad=True) # Placeholder such that the code runs 
