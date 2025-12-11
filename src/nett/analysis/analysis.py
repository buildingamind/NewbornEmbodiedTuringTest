"""Analyze the results of a run."""

from pathlib import Path
from typing import Optional
from .utils import merge, train_viz, test_viz

def analyze(
    experiment: str,
    run_dir: str,
    output_dir: Optional[str] = None,
    bar_order: str | list[int] = "default",
    color_bars: bool = True,
) -> None:
    """
    Analyze the results of a run.

    This method is a static method and does not require an instance of the NETT class to be called.

    Args:
        experiment (str): The  experiment to be analyzed. It can be "parsing", "binding", "viewinvariant", "biomotion", "transitionalprobabilities", "facerecognition", "identityandview", "slowness", or "smoothness".
        run_dir (str | Path): The directory where the run results are stored.
        output_dir (str | Path, optional): The directory where the analysis results will be stored.
            If None, the analysis results will be stored in the run directory.
        bar_order (str | list[int], optional): The order in which the bars are to be displayed in the analysis plots.
            Default is "default". Can be "default", "asc", "desc", or a list of bar numbers (e.g. [3,1,2,4]).
        color_bars (bool, optional): Whether to color the bars in the analysis plots by condition. Default is True.

    Returns:
        None

    Example:
        >>> nett.analyze("parsing", run_dir="./test_run", output_dir="./results") # benchmarks is an instance of NETT
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
    train_viz(output_dir, output_dir)

    # test
    print("Running analysis for [test]")
    test_viz(output_dir, chick_data_dir, bar_order_str, color_bars)

    print(f"Analysis complete. See results at {output_dir}")
