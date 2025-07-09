import torch
import torch.nn as nn

# import torchvision
from torchvision import models, transforms, utils

# from torch.autograd import Variable
# import numpy as np
import matplotlib.pyplot as plt

# import scipy.misc
from PIL import Image

# import json
import os
import glob
from stable_baselines3 import PPO


def _visualize_and_save_features(model, image_path, output_dir):
    """
    Generates and saves feature map visualizations for a given model and image.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=1.0),
        ]
    )

    image = Image.open(image_path)
    # plt.imshow(image) # Displaying image is commented out for batch processing

    # we will save the conv layer weights in this list
    model_weights = []
    # we will save the conv layers in this list
    conv_layers = []
    # get all the model children as list
    # The policy network is what we're interested in.
    model_children = list(model.policy.features_extractor.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    # Assuming at most 20 layers to visualize based on original code's subplot grid
    num_layers = len(processed)
    rows = (num_layers + 3) // 4
    for i in range(num_layers):
        a = fig.add_subplot(rows, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split("(")[0], fontsize=30)

    image_name = os.path.basename(image_path)
    output_filename = f"feature_maps_{os.path.splitext(image_name)[0]}.jpg"
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory


def feature_visualization(input_path, output_path, images_path):
    """
    feature visualization function to find models, create output directories, and process images.
    """
    model_paths = glob.glob(
        os.path.join(input_path, "*", "brain_*", "model", "latest_model.zip"),
        recursive=True,
    )
    image_paths = glob.glob(os.path.join(images_path, "*.png"))

    if not model_paths:
        print(f"No models found in {input_path}")
        return
    if not image_paths:
        print(f"No images found in {images_path}")
        return

    for model_path in model_paths:
        try:
            # Extract imprint condition and brain_id from path
            parts = model_path.split(os.sep)
            imprint_condition = parts[-5]
            brain_id = parts[-4]

            # Create corresponding output directory
            current_output_dir = os.path.join(output_path, imprint_condition, brain_id)
            os.makedirs(current_output_dir, exist_ok=True)

            print(f"Loading model: {model_path}")
            model = PPO.load(model_path)

            for image_path in image_paths:
                print(f"Processing image: {image_path}")
                _visualize_and_save_features(model, image_path, current_output_dir)

            print(f"Finished processing for model {brain_id} under {imprint_condition}")

        except Exception as e:
            print(f"Could not process model {model_path}. Error: {e}")
