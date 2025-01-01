import os
import torch
import torch.optim as optim

from helper_functions import *
from your_code_here import *

def run_style_transfer(vgg_mean, vgg_std, content_img, style_img, num_steps, random_init, w_style, w_content, w_tv, w_edge, print_iter=50):
    """ Neural Style Transfer optmization procedure for a single style image.
    
    # Parameters:
        @vgg_mean, VGG channel-wise mean, torch.tensor of size (c)
        @vgg_std, VGG channel-wise standard deviation, detorch.tensor of size (c)
        @content_img, torch.tensor of size (1, c, h, w)
        @style_img, torch.tensor of size (1, c, h, w)
        @num_steps, int, iteration steps
        @random_init, bool, whether to start optimizing with based on a random image. If false,
            the content image is as initialization.
        @w_style, float, weight for style loss
        @w_content, float, weight for content loss 
        @w_tv, float, weight for total variation loss
        @w_edge, float, weight for edge loss
        @print_iter, int, iteration interval for printing the losses

    # Returns the style-transferred image
    """

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # TODO: 1. Normalize Input images
    normed_style_img = normalize(style_img, vgg_mean, vgg_std)
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    if w_edge > 0:
        target_gradient_img = get_gradient_imgs(to_grayscale(normed_content_img)).detach()

    # Retrieve feature maps for content and style image
    # We do not need to calculate gradients for these feature maps
    with torch.no_grad():
        style_features = model(normed_style_img)
        content_features = model(normed_content_img)
    
    # Either initialize the image from random noise or from the content image
    if random_init:
        optim_img = torch.randn(content_img.data.size(), device=device)
        optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
    else:
        optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

    # Initialize optimizer and set image as parameter to be optimized
    optimizer = optim.LBFGS([optim_img])
    
    # Training Loop
    iter = [0]
    while iter[0] <= num_steps:

        def closure():
            
            # Set gradients to zero before next optimization step
            optimizer.zero_grad()

            # Clamp image to lie in correct range
            with torch.no_grad():
                optim_img.clamp_(0, 1)

            # Retrieve features of image that is being optimized
            normed_img = normalize(optim_img, vgg_mean, vgg_std)
            input_features = model(normed_img)

            loss = torch.tensor([0.], device=device)
            # TODO: 2. Calculate the content loss
            c_loss = 0
            if w_content > 0:
                c_loss = w_content * content_loss(input_features, content_features, content_layers)

            # TODO: 3. Calculate the style loss
            s_loss = 0
            if w_style > 0:
                s_loss = w_style * style_loss(input_features, style_features, style_layers)
            
            # TODO: 4. Calculate the total variation loss
            tv_loss = 0
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)

            e_loss = 0
            if w_edge > 0:
                # TODO: 5. Calculate the gradient images based on the sobel kernel
                gradient_optim = get_gradient_imgs(to_grayscale(optim_img))
                # TODO: 6. Calculate the edge loss 
                e_loss = w_edge * edge_loss(target_gradient_img, gradient_optim)
                

            # Sum up the losses and do a backward pass
            loss = loss + s_loss + c_loss + tv_loss + e_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % print_iter == 0:
                print(f'iter {iter[0]}: Content Loss: {c_loss.item():4f} | Style Loss: {s_loss.item():4f} | TV Loss: {tv_loss.item():4f} | Edge Loss: {e_loss.item():4f}')

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)
    
    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    return optim_img, target_gradient_img

if __name__ == '__main__':
    seed_everything(101)

    device = 'cpu' # NOTE: Make sure that if you use cuda that it also runs on CPU
    style_img_name = 'gogh.jpg' # 'gogh.jpg', 'munch.jpg', 'duck.jpg'
    content_img_name = 'duck.jpg' # 'duck.jpg', 'gogh.jpg', 'munch.jpg'
    img_size = 128 # '128', '256'
    print_iter = 50

    # Hyperparameters
    # Sets of hyperparameters that worked well for us
    # NOTE: For debugging purposes, you can set num_steps to a lower number
    if img_size == 128:
        num_steps = 400
        w_content = 2
        w_style = 1e5
        w_tv = 2e1
        w_edge = 2e1
    else:
        num_steps = 600
        w_style = 5e5
        w_content = 1
        w_tv = 2e1
        w_edge = 2e1

    # Choose what feature maps to extract for the content and style loss
    # We use the ones as mentioned in Gatys et al. 2016
    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # Paths
    out_folder = 'outputs'
    style_img_path = os.path.join('data', style_img_name) 
    content_img_path = os.path.join('data', content_img_name)
    os.makedirs(out_folder, exist_ok=True)

    # Load style and content images as resized (spatially square) tensors
    style_img = image_loader(style_img_path, device=device, img_size=img_size)
    content_img = image_loader(content_img_path, device=device, img_size=img_size)

    # Define the channel-wise mean and standard deviation used for VGG training
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Image optimization
    output, target_gradient_img = run_style_transfer(
        vgg_mean, vgg_std, content_img, style_img, num_steps=num_steps, 
        random_init=True, w_style=w_style, w_content=w_content, w_tv=w_tv, 
        w_edge=w_edge, print_iter=print_iter)

    file_name = f"{content_img_name.split('.')[0]}_to_{style_img_name.split('.')[0]}"
    output_name1 = f'{file_name} img_size-{img_size} num_steps-{num_steps} w_content-{w_content} w_style-{w_style} w_tv-{w_tv} w_edge-{w_edge} '
    save_image(output, title=output_name1, out_folder=out_folder)
    
    # Displaying the gradient images
    Gx = target_gradient_img[:,0]
    Gx = (Gx - Gx.min()) / (Gx.max() - Gx.min())
    Gy = target_gradient_img[:,1]
    Gy = (Gy - Gy.min()) / (Gy.max() - Gy.min())
    save_image(Gx, title=f'{content_img_name.split(".")[0]} sobel_x', out_folder=out_folder)
    save_image(Gy, title=f'{content_img_name.split(".")[0]} sobel_y', out_folder=out_folder)
