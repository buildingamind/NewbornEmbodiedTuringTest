import argparse
import os
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
from stable_baselines3 import PPO
import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.save_util"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="stable_baselines3.common.save_util"
)


def _get_model_params(model_path):
    """Extracts and flattens model parameters from a file.

    Args:
        model_path (str): The path to the model file.
            Supported formats: .pth, .pkl, .zip.

    Returns:
        np.ndarray: A 1D numpy array of the model's parameters.
            Returns an empty array if parameters cannot be extracted.
    """
    if model_path.endswith(".pth"):
        model_data = torch.load(model_path, map_location=torch.device("cpu"))
        if isinstance(model_data, dict) and "state_dict" in model_data:
            params = model_data["state_dict"]
        else:
            params = model_data

        all_params = []
        for param in params.values():
            if isinstance(param, torch.Tensor):
                all_params.append(param.cpu().numpy().flatten())
        return np.concatenate(all_params) if all_params else np.array([])

    elif model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        all_params = []
        if hasattr(model, "parameters"):
            for param in model.parameters():
                all_params.append(param.data.cpu().numpy().flatten())
        elif hasattr(model, "coef_"):
            all_params.append(model.coef_.flatten())
        elif hasattr(model, "feature_importances_"):
            all_params.append(model.feature_importances_.flatten())

        return np.concatenate(all_params) if all_params else np.array([])

    elif model_path.endswith(".zip"):
        model = PPO.load(model_path, device="cpu")
        params = model.policy.state_dict()
        all_params = []
        for param in params.values():
            if isinstance(param, torch.Tensor):
                all_params.append(param.cpu().numpy().flatten())
        return np.concatenate(all_params) if all_params else np.array([])
    else:
        return np.array([])


def _create_tsne_plot(data, output_path, title):
    """Creates and saves a t-SNE plot.

    Args:
        data (np.ndarray): The data to be visualized.
        output_path (str): The path to save the generated plot.
        title (str): The title for the plot.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(data) < 2:
        return
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, max(1, len(data) - 1)),
    )
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig(output_path)
    plt.close()


def generate_tSNEs(input_dir, output_dir):
    """Generates t-SNE diagrams for model weights for a batch of models.

    This function walks through a directory of models structured by imprinting
    condition and brain ID, generates a t-SNE plot for the weights of each
    model file found, and saves the plots to a corresponding directory
    structure in the output directory.

    Args:
        input_dir (str): The path to the input directory. It should contain
            subdirectories for each imprinting condition.
        output_dir (str): The path to the output directory where the t-SNE
            plots will be saved.
    """
    for imprinting_condition in os.listdir(input_dir):
        condition_path = os.path.join(input_dir, imprinting_condition)
        if not os.path.isdir(condition_path):
            continue

        for brain_dir in os.listdir(condition_path):
            if not brain_dir.startswith("brain_"):
                continue

            brain_path = os.path.join(condition_path, brain_dir)
            model_dir = os.path.join(brain_path, "model")

            if not os.path.isdir(model_dir):
                continue

            output_brain_dir = os.path.join(output_dir, imprinting_condition, brain_dir)
            os.makedirs(output_brain_dir, exist_ok=True)

            for model_file in [
                "feature_extractor.pth",
                "policy.pkl",
                "latest_model.zip",
            ]:
                model_path = os.path.join(model_dir, model_file)
                if os.path.exists(model_path):
                    print(f"Processing {model_path}...")
                    params = _get_model_params(model_path)

                    if params.size > 0:
                        # For t-SNE on weights, we need to decide what to visualize.
                        # A t-SNE of a single model's weights isn't standard.
                        # A possible interpretation is to visualize the distribution of weights within the model.
                        # Here, we'll treat each weight as a data point.

                        # To make the plot manageable, we can sample the weights if there are too many.
                        if params.size > 5000:
                            indices = np.random.choice(params.size, 5000, replace=False)
                            sampled_params = params[indices]
                        else:
                            sampled_params = params

                        output_filename = f"{os.path.splitext(model_file)[0]}_tsne.png"
                        output_filepath = os.path.join(
                            output_brain_dir, output_filename
                        )

                        plot_title = f"t-SNE of {model_file} weights\n({imprinting_condition} - {brain_dir})"
                        _create_tsne_plot(
                            sampled_params.reshape(-1, 1), output_filepath, plot_title
                        )
                        print(f"Saved t-SNE plot to {output_filepath}")
                    else:
                        print(f"Could not extract parameters from {model_path}")
