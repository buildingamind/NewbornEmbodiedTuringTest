import os
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image, ImageChops
from .utils import merge, train_viz, test_viz


def dst(run_dir: str | Path, output_dir: str | Path) -> None:
    try:
        # TODO may need to clean up this file structure
        # set paths
        run_dir = Path(run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        rec_path = Path.joinpath(run_dir, "recordings", "states")

        if not rec_path.exists():
            raise FileNotFoundError(f"Recording directory {rec_path} does not exist.")

        obs = np.loadtxt(Path.joinpath(rec_path, "obs.txt"), dtype=int, ndmin=2)
        actions = np.loadtxt(
            Path.joinpath(rec_path, "actions.txt"), dtype=float, ndmin=2
        )
        if Path.joinpath(rec_path, "states.txt").exists():
            states = np.loadtxt(
                Path.joinpath(rec_path, "states.txt"), dtype=float, ndmin=2
            )
        else:
            states = None

        # Normalize data
        obs = (np.array(obs) - np.min(obs)) / (np.max(obs) - np.min(obs))
        actions = (np.array(actions) - np.min(actions)) / (
            np.max(actions) - np.min(actions)
        )
        if states is not None:
            states = (np.array(states) - np.min(states)) / (
                np.max(states) - np.min(states)
            )

        # perform PCA on observations
        pca1 = PCA(n_components=1)
        pca2 = PCA(n_components=2)
        pc_obs = pca2.fit_transform(obs)
        pc_actions = pca1.fit_transform(actions)
        pc_states = pca2.fit_transform(states) if states is not None else None

        ax = plt.figure().add_subplot(projection="3d")

        ax.plot(*np.hstack((pc_obs, pc_actions)).T)

        ax.set_xlabel("PC Observation 1")
        ax.set_ylabel("PC Observation 2")
        ax.set_zlabel("PC Behavior")
        ax.figure.savefig(output_dir.joinpath("trajectories.png"))

    except Exception as e:
        raise f"Error in dst: {e}"


def timelapse(data_dir: Path | str, output_dir: Path | str):

    try:
        # Ensure data_dir and output_dir are Path objects
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        # Create the 'paths' directory inside output_dir
        paths_dir = output_dir / "paths"
        paths_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over directories matching data_dir / (*) / "brain_(*)"
        for brain_dir in data_dir.glob("*/*"):
            if brain_dir.is_dir() and brain_dir.name.startswith("brain_"):
                # Get a list of all PNG images in the directory
                images = []
                recording_dir = brain_dir / "recordings" / "chamber"
                if not recording_dir.exists():
                    print(
                        f"Skipping {brain_dir} as it does not contain a ChamberRecorder directory"
                    )
                    continue
                for f in os.listdir(recording_dir):
                    full_path = os.path.join(recording_dir, f)
                    if os.path.isfile(full_path) and f.lower().endswith(".png"):
                        images.append(full_path)

                # Check if there are at least two images to blend
                if len(images) < 2:
                    print(f"Not enough images to blend for {brain_dir}. Skipping.")
                    continue

                # Open the first image
                result_image = Image.open(images[0]).convert("RGBA")

                # Loop through each image and blend it with the accumulated result
                for img_path in images[1:]:
                    img = Image.open(img_path).convert("RGBA")
                    result_image = ImageChops.lighter(result_image, img)

                # Extract the wildcard captures
                condition = brain_dir.parent.name
                brain_num = brain_dir.name[len("brain_") :]
                # Create the filename and the empty file
                filename = f"{condition}{brain_num}"

                # Save the final blended image to the desired output filename
                result_image.save(paths_dir / (filename + ".png"))

                print(f"{filename} completed")
    except Exception as e:
        raise f"Error in timelapse: {e}"


# TODO v0.3, make .analyze() a staticmethod so that it does not need a class instance to call
# TODO v0.3. add support for user specified output_dir
# Discussion v0.3 is print okay or should we have it log using nett's logger?
# Discussion v0.3 move this out of the class entirely? from nett import analyze, analyze(...)


# TODO: Add option to not have a config here either?
def analyze(
    experiment: str,
    run_dir: str,
    output_dir: Optional[str] = None,
    ep_bucket: int = 100,
    num_episodes: int = 1000,
    bar_order: str | list[int] = "default",
    color_bars: bool = True,
) -> None:
    """
    Analyze the results of a run.

    This method is a static method and does not require an instance of the NETT class to be called.

    Args:
        experiment (str): The  experiment to be analyzed. It can be "parsing", "binding", "viewinvariance", "facerecognition", "biomotion", or "statisticallearning".
        run_dir (str | Path): The directory where the run results are stored.
        output_dir (str | Path, optional): The directory where the analysis results will be stored.
            If None, the analysis results will be stored in the run directory.
        ep_bucket (int, optional): The number of episodes to be grouped together for analysis.
        num_episodes (int, optional): The number of episodes to be analyzed.
        bar_order (str | list[int], optional): The order in which the bars are to be displayed in the analysis plots.
            Default is "default". Can be "default", "asc", "desc", or a list of bar numbers (e.g. [3,1,2,4]).
        color_bars (bool, optional): Whether to color the bars in the analysis plots by condition. Default is True.

    Returns:
        None

    Example:
        >>> nett.analyze(run_dir="./test_run", output_dir="./results") # benchmarks is an instance of NETT
    """
    # TODO may need to clean up this file structure
    # set paths
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    analysis_dir = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = run_dir.joinpath("results")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chick_data_dir = analysis_dir / "ChickData" / f"{experiment.lower()}.csv"

    if not chick_data_dir.exists():
        chick_data_dir = None
        print(f"Warning: No chick data exists for '{experiment}'.")
    elif not run_dir.exists():
        raise ValueError(f"'{run_dir}' is not a valid run directory.")
    elif not analysis_dir.exists():
        raise ValueError(
            f"'{analysis_dir}' is not a valid analysis directory. This is likely an error in the package."
        )

    # translate bar_order for R to read
    bar_order_str = str(bar_order).translate(
        {ord(i): None for i in " []"}
    )  # remove spaces and brackets from bar_order

    # merge
    print("Running merge")
    merge(run_dir, output_dir)

    print("Running analysis for [train]")
    train_viz(output_dir, output_dir, ep_bucket, num_episodes)

    # test
    print("Running analysis for [test]")
    test_viz(output_dir, chick_data_dir, bar_order_str, color_bars)

    print(f"Analysis complete. See results at {output_dir}")
