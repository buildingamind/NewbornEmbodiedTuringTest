# Code taken from https://github.com/Nguyen-Hoa/Activation-Maximization and modified for nett
import torch
import os
import glob
import argparse

# Showing images
import cv2

# Reading images
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile

# Gaussian Kernel
from scipy.ndimage import gaussian_filter

from stable_baselines3 import PPO

# https://medium.com/analytics-vidhya/deep-dream-visualizing-the-features-learnt-by-convolutional-networks-in-pytorch-b7296ae3b7f
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def _image_converter(im):

    # move the image to cpu
    im_copy = im.cpu()

    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1, 2, 0)

    # clip negative values as plt.imshow() only accepts
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1)

    return im_copy


# https://stackoverflow.com/questions/50420168/how-do-i-load-up-an-image-and-convert-it-to-a-proper-tensor-for-pytorch
def _get_image(img_path):
    img = Image.open(img_path)  # use pillow to open a file
    img = img.resize((256, 256))  # resize the file to 256x256
    img = img.convert("RGB")  # convert image to RGB channel

    img = asarray(img).transpose(
        -1, 0, 1
    )  # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
    img = img / 255
    img = torch.from_numpy(img)  # create the image tensor
    return img


"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""


def _layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output

    return hook


"""
Reguarlizer, crop by absolute value of pixel contribution
"""


def _abs_contrib_crop(img, threshold=0):

    abs_img = torch.abs(img)
    smalls = abs_img < percentile(abs_img, threshold)

    return img - img * smalls


"""
Regularizer, crop if norm of pixel values below threshold
"""


def _norm_crop(img, threshold=0):

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < percentile(norm, threshold)
    smalls = tile(smalls, (3, 1, 1))

    # Crop pixels from image
    crop = img - img * smalls
    return crop


"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""


def _act_max(
    network,
    input,
    layer_activation,
    layer_name,
    unit,
    steps=5,
    alpha=torch.tensor(100),
    generate_gif=False,
    path_to_gif="./",
    L2_Decay=False,
    theta_decay=0.1,
    Gaussian_Blur=False,
    theta_every=4,
    theta_width=1,
    verbose=False,
    Norm_Crop=False,
    theta_n_crop=30,
    Contrib_Crop=False,
    theta_c_crop=30,
):

    best_activation = -float("inf")
    best_img = input.clone().detach()

    for k in range(steps):

        input.retain_grad()  # non-leaf tensor
        # network.zero_grad()

        # Propogate image through network,
        # then access activation of target layer
        network(input)
        layer_out = layer_activation[layer_name]

        # compute gradients w.r.t. target unit,
        # then access the gradient of input (image) w.r.t. target unit (neuron)
        if layer_out[0].ndim == 1:
            target_activation = layer_out[0][unit]
        else:
            target_activation = layer_out[0][unit].mean()

        target_activation.backward(retain_graph=True)
        img_grad = input.grad

        # Gradient Step
        # input = input + alpha * dimage_dneuron
        input = torch.add(input, torch.mul(img_grad, alpha))

        # regularization does not contribute towards gradient
        """
        DEV:
            Detach input here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input = torch.mul(input, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every == 0:
                temp = input.squeeze(0)
                temp = temp.detach().numpy()
                for channel in range(3):
                    cimg = gaussian_filter(temp[channel], theta_width)
                    temp[channel] = cimg
                temp = torch.from_numpy(temp)
                input = temp.unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                input = _norm_crop(input.detach().squeeze(0), threshold=theta_n_crop)
                input = input.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                input = _abs_contrib_crop(
                    input.detach().squeeze(0), threshold=theta_c_crop
                )
                input = input.unsqueeze(0)

        input.requires_grad_(True)

        if verbose:
            print("step: ", k, "activation: ", target_activation.item())

        if generate_gif:
            frame = input.detach().squeeze(0)
            frame = _image_converter(frame)
            frame = frame * 255
            cv2.imwrite(os.path.join(path_to_gif, f"{k}.jpg"), frame)

        # Keep highest activation
        if best_activation < target_activation.item():
            best_activation = target_activation.item()
            best_img = input.clone().detach()

    return best_img


"""
Prepare Input from Image
"""


def _load_image(path_to_image, device=False):
    tensor_image = _get_image(path_to_image)
    tensor_image = normalize(tensor_image)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image.requires_grad = True
    if device:
        tensor_image = tensor_image.type(torch.cuda.FloatTensor)
    else:
        tensor_image = tensor_image.type(torch.FloatTensor)
    return tensor_image


"""
Prepare Dummy Input
"""


# TODO: Use real images instead of dummy images
def _load_dummy_image(device=False):
    dummy_image = torch.randn(3, 256, 256, requires_grad=True)
    dummy_image = normalize(dummy_image)
    if device:
        dummy_image = dummy_image.type(torch.cuda.FloatTensor)
    else:
        dummy_image = dummy_image.type(torch.FloatTensor)
    dummy_image = dummy_image.unsqueeze(0)
    return dummy_image


def run_activation_maximization(input_path, output_path):
    """
    Finds all models in input_path, runs activation maximization, and saves results to output_path.
    """
    model_paths = glob.glob(
        os.path.join(input_path, "**", "latest_model.zip"), recursive=True
    )
    print(f"Found {len(model_paths)} models.")

    for model_path in model_paths:
        print(f"Processing model: {model_path}")

        # Create output directory structure
        rel_path = os.path.relpath(os.path.dirname(model_path), input_path)
        # Go up from .../model/
        imprint_condition_brain_id = os.path.dirname(rel_path)

        current_output_dir = os.path.join(output_path, imprint_condition_brain_id)
        os.makedirs(current_output_dir, exist_ok=True)
        print(f"Output will be saved to: {current_output_dir}")

        # Load model
        try:
            agent = PPO.load(model_path)
            model = agent.policy
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        # Prepare for activation maximization
        activations = {}

        # Hook into a layer
        # Example: hook into the output of the CNN feature extractor
        try:
            target_layer = model.features_extractor.cnn
            layer_name = "features_extractor.cnn"
            target_layer.register_forward_hook(_layer_hook(activations, layer_name))
        except AttributeError:
            print(
                "Could not find 'model.features_extractor.cnn'. Trying 'model.features_extractor'."
            )
            try:
                target_layer = model.features_extractor
                layer_name = "features_extractor"
                target_layer.register_forward_hook(_layer_hook(activations, layer_name))
            except AttributeError:
                print("Could not find a suitable layer to hook into. Skipping model.")
                continue

        # Get dummy input
        input_image = _load_dummy_image()

        # Run activation maximization for a few units
        # To get the number of units, we need a forward pass
        with torch.no_grad():
            output = target_layer(
                model.features_extractor.obs_to_tensor(input_image)[0]
            )
            num_units = output.shape[1]

        units_to_visualize = min(num_units, 10)  # Visualize first 10 units or less

        for unit_idx in range(units_to_visualize):
            print(f"Maximizing activation for unit {unit_idx}")
            unit_output_dir = os.path.join(current_output_dir, f"unit_{unit_idx}")
            os.makedirs(unit_output_dir, exist_ok=True)

            _act_max(
                network=model,
                input=input_image.clone().detach().requires_grad_(True),
                layer_activation=activations,
                layer_name=layer_name,
                unit=unit_idx,
                steps=50,
                alpha=torch.tensor(0.1),
                path_to_gif=unit_output_dir,
                generate_gif=True,
                verbose=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run activation maximization on trained models."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Root directory of models to process.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save output images.",
    )
    args = parser.parse_args()

    run_activation_maximization(args.input_path, args.output_path)
