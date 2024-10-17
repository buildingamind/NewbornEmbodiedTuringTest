"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
   :synopsis: Main class for training, testing and analyzing brains in environments.

"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy
from itertools import product
from concurrent.futures import ProcessPoolExecutor, Future, wait as future_wait, FIRST_COMPLETED
from PIL import Image, ImageChops

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mlagents_envs.exception import UnityWorkerInUseException
from sb3_contrib import RecurrentPPO
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from stable_baselines3.common.env_checker import check_env
import yaml

from stable_baselines3.common.vec_env import SubprocVecEnv

from nett.utils.io import mute
from nett.utils.job import Job
from nett.utils.environment import port_in_use

from nett.brain.builder import Brain
from nett.body.builder import Body
from nett.environment.builder import Environment

from nett.fast2 import fast as fastrun

class NETT:
    """
    The NETT class is the main class for training, testing, and analyzing brains in environments.

    Args:
        brain (Brain): The brain to be trained and tested.
        body (Body): The body to be used for training and testing the brain.
        environment (Environment): The environment in which the brain is to be trained and tested.

    Example:
        >>> from nett import NETT
        >>> # create a brain, body, and environment
        >>> benchmarks = NETT(brain, body, environment)
    """

    def __init__(self, 
                 brain: "nett.Brain" = None, 
                 body: "nett.Body" = None, 
                 environment: "nett.Environment" = None, 
                 config: Path | str | list[Path | str] = None, 
                 fast: bool = False) -> None:
        """
        Initialize the NETT class.
        """

        # for NVIDIA memory management
        nvmlInit()

        if fast and config is not None:
            fastrun(config)

        # initialize logger
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)

        if config is not None:
            try:
                if isinstance(config, list):
                    raise NotImplementedError("Multiple config files are not supported yet.")
                else:
                    with open(config, "r") as file:
                        config_text = yaml.safe_load(file)
                    self.brain = Brain(**config_text["Brain"])
                    self.body = Body(**config_text["Body"])
                    self.environment = Environment(**config_text["Environment"])
                    if "Run" in config_text:
                        self.run(**config_text["Run"])
            except Exception as e:
                self.logger.exception("Error in loading config")
                raise e
        else:
            self.brain = brain
            self.body = body
            self.environment = environment        


    def run(self,
            output_dir: Path | str,
            num_brains: int = 1,
            mode: str = "full",
            train_eps: int = 1000,
            test_eps: int = 20,
            batch_mode: bool = True,
            devices: Optional[list[int]] =  None,
            job_memory: str | int = 4,
            buffer: float = 1.2,
            steps_per_episode: int = 1000,
            conditions: Optional[list[str]] = None,
            verbose: int = True,
            synchronous: bool = False,
            save_checkpoints: bool = False,
            checkpoint_freq: int = 30_000,
            record: Optional[list[str]] = [],
            recording_eps: int = 10,
            base_port: int = 5004) -> list[Future]:
        """
        Run the training and testing of the brains in the environment.

        Args:
            output_dir (Path | str): The directory where the run results will be stored.
            num_brains (int, optional): The number of brains to be trained and tested. Defaults to 1.
            mode (str, optional): The mode in which the brains are to be trained and tested. It can be "train", "test", or "full". Defaults to "full".
            train_eps (int, optional): The number of episodes the brains are to be trained for. Defaults to 1000.
            test_eps (int, optional): The number of episodes the brains are to be tested for. Defaults to 20.
            batch_mode (bool, optional): Whether to run in batch mode, which will not display Unity windows. Good for headless servers. Defaults to True.
            devices (list[int], optional): The list of devices to be used for training and testing. If None, all available devices will be used. Defaults to None.
            job_memory (int, optional): The memory allocated, in Gigabytes, for a single job. Defaults to 4.
            buffer (float, optional): The buffer for memory allocation. Defaults to 1.2.
            steps_per_episode (int, optional): The number of steps per episode. Defaults to 1000.
            verbose (int, optional): Whether or not to print info statements. Defaults to True.
            synchronous (bool, optional): Whether to wait for all jobs to end rather than return a Promise. Defaults to False.
            save_checkpoints (bool, optional): Whether to save checkpoints during training. Defaults to False.
            checkpoint_freq (int, optional): The frequency at which checkpoints are saved. Defaults to 30_000.
            record (list[str], optional): The list of what record options to use. Can include "agent" for recording the agent's view, "chamber" for recording the top-down view of the chamber, and "state" for recording the observations, actions, and states.
            recording_eps (int, optional): Number of episodes to record for. Defaults to 10.
            base_port (int, optional): The base port number to use for communication with the Unity environment. Defaults to 5004.

        Returns:
            list[Future]: A list of futures representing the jobs that have been launched.

        Example:
            >>> job_sheet = benchmarks.run(output_dir="./test_run", num_brains=2, train_eps=100, test_eps=10) # benchmarks is an instance of NETT
        """
        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {output_dir.resolve()}")

        # calculate iterations
        iterations: dict[str, int] = {
            "train": steps_per_episode * train_eps, #10*5 = 50
            "test": self.environment.num_test_conditions
        }

        if not issubclass(self.brain.algorithm, RecurrentPPO):
            iterations["test"] *= steps_per_episode

        # initialize job object
        Job.initialize(
            mode=mode,
            output_dir=output_dir,
            save_checkpoints=save_checkpoints, 
            steps_per_episode=steps_per_episode,
            checkpoint_freq=checkpoint_freq,
            reward=self.brain.reward,
            batch_mode=batch_mode, 
            iterations=iterations,
            record=record,
            recording_eps=recording_eps,
            test_eps=test_eps
            )

        # validate devices
        devices = self._validate_devices(devices)
        self.logger.info(f"Devices that will be used: {devices}")

        # estimate memory for a single job
        if job_memory == "auto":
            self.logger.info("Estimating Job Memory...")
            job_memory = int(buffer * self._estimate_job_memory(devices, base_port))
            self.logger.info(f"Estimated Job Memory: {job_memory / (1024**3)} GiB")
        else:
            job_memory *= buffer * 1024 * 1024 * 1024 # set memory to be in GiB
        

        # get task set
        task_set: set[tuple[str,int]] = self._get_task_set(num_brains, self.environment.imprinting_conditions, conditions)
        
        # schedule jobs
        jobs, waitlist = self._schedule_jobs(task_set, devices, job_memory, base_port, self.logger)
        self.logger.info("Scheduled jobs")

        # launch jobs
        self.logger.info("Launching")
        job_sheet = self._launch_jobs(jobs, synchronous, waitlist, verbose)

        # return control back to the user after launching jobs, do not block
        return job_sheet

    def status(self, job_sheet: dict[Future, Job]) -> pd.DataFrame:
        """
        Get the status of the jobs in the job sheet.

        Args:
            job_sheet (dict[Future, Job]): The job sheet returned by the .launch_jobs() method.

        Returns:
            pd.DataFrame: A dataframe containing the status of the jobs in the job sheet.

        Example:
            >>> status = benchmarks.status(job_sheet)
            >>> # benchmarks is an instance of NETT, job_sheet is the job sheet returned by the .run() method
        """
        selected_columns = ["brain_id", "condition", "device"]
        filtered_job_sheet = self._filter_job_sheet(job_sheet, selected_columns)
        return pd.json_normalize(filtered_job_sheet)
    
    @staticmethod
    def dst(run_dir: str | Path,
            output_dir: str | Path) -> None:
        try:
            # TODO may need to clean up this file structure
            # set paths
            run_dir = Path(run_dir).resolve()
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            rec_path = Path.joinpath(run_dir,"env_recs", "states")

            if not rec_path.exists():
                raise FileNotFoundError(f"Recording directory {rec_path} does not exist.")
            
            obs = np.loadtxt(Path.joinpath(rec_path, "obs.txt"), dtype=int, ndmin=2)
            actions = np.loadtxt(Path.joinpath(rec_path, "actions.txt"), dtype=float, ndmin=2)
            if (Path.joinpath(rec_path, "states.txt").exists()):
                states = np.loadtxt(Path.joinpath(rec_path, "states.txt"), dtype=float, ndmin=2)
            else:
                states = None

            # Normalize data
            obs = (np.array(obs)-np.min(obs))/(np.max(obs)-np.min(obs))
            actions = (np.array(actions)-np.min(actions))/(np.max(actions)-np.min(actions))
            if states is not None:
                states = (np.array(states)-np.min(states))/(np.max(states)-np.min(states))
            
            # perform PCA on observations
            from sklearn.decomposition import PCA
            pca1 = PCA(n_components=1)
            pca2 = PCA(n_components=2)
            pc_obs = pca2.fit_transform(obs)
            pc_actions = pca1.fit_transform(actions)
            pc_states = pca2.fit_transform(states) if states is not None else None

            ax = plt.figure().add_subplot(projection='3d')

            ax.plot(*np.hstack((pc_obs, pc_actions)).T)

            ax.set_xlabel('PC Observation 1')
            ax.set_ylabel('PC Observation 2')
            ax.set_zlabel('PC Behavior')
            ax.figure.savefig(output_dir.joinpath("trajectories.png"))

        except Exception as e:
            raise f"Error in dst: {e}"

    @staticmethod
    def timelapse(data_dir: Path | str, output_dir: Path | str):

        try:
            # Ensure data_dir and output_dir are Path objects
            data_dir = Path(data_dir)
            output_dir = Path(output_dir)

            # Create the 'paths' directory inside output_dir
            paths_dir = output_dir / 'paths'
            paths_dir.mkdir(parents=True, exist_ok=True)

            # Iterate over directories matching data_dir / (*) / "brain_(*)"
            for brain_dir in data_dir.glob('*/*'):
                if brain_dir.is_dir() and brain_dir.name.startswith('brain_'):
                    # Get a list of all PNG images in the directory
                    images = []
                    recording_dir = brain_dir / 'env_recs' / 'ChamberRecorder'
                    if not recording_dir.exists():
                        print(f"Skipping {brain_dir} as it does not contain a ChamberRecorder directory")
                        continue
                    for f in os.listdir(recording_dir):
                        full_path = os.path.join(recording_dir, f)
                        if os.path.isfile(full_path) and f.lower().endswith('.png'):
                            images.append(full_path)

                    # Check if there are at least two images to blend
                    if len(images) < 2:
                        print("Not enough images to blend.")
                        sys.exit(1)

                    # Open the first image
                    result_image = Image.open(images[0]).convert('RGBA')

                    # Loop through each image and blend it with the accumulated result
                    for img_path in images[1:]:
                        img = Image.open(img_path).convert('RGBA')
                        result_image = ImageChops.lighter(result_image, img)

                    # Extract the wildcard captures
                    condition = brain_dir.parent.name
                    brain_num = brain_dir.name[len('brain_'):]
                    # Create the filename and the empty file
                    filename = f"{condition}{brain_num}"

                    # Save the final blended image to the desired output filename
                    result_image.save(paths_dir / (filename+".png"))

                    print(f"{filename} completed")
        except Exception as e:
            raise f"Error in timelapse: {e}"

    # TODO v0.3, make .analyze() a staticmethod so that it does not need a class instance to call
    # TODO v0.3. add support for user specified output_dir
    # Discussion v0.3 is print okay or should we have it log using nett's logger?
    # Discussion v0.3 move this out of the class entirely? from nett import analyze, analyze(...)

    # TODO: Add option to not have a config here either?

    @staticmethod
    def analyzePython(config: str,
                run_dir: str | Path,
                output_dir: Optional[str | Path] = None,
                ep_bucket: int = 100,
                num_episodes: int = 1000,
                bar_order: str | list[int] = "default",
                color_bars: bool = True) -> None:
        """
        Analyze the results of a run.

        This method is a static method and does not require an instance of the NETT class to be called.

        Args:
            config (str): The configuration of the experiment to be analyzed. It can be "parsing", "binding", "viewinvariant", "facedifferentiation", "biomotion", or "statisticallearning".
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

        analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
        if output_dir is None:
            output_dir = run_dir.joinpath("results")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        chick_data_dir = Path(analysis_dir).joinpath("ChickData", f"{config.lower()}.csv")

        if not chick_data_dir.exists():
            raise ValueError(f"'{config}' is not a valid config.")
        elif not run_dir.exists():
            raise ValueError(f"'{run_dir}' is not a valid run directory.")
        elif not analysis_dir.exists():
            raise ValueError(f"'{analysis_dir}' is not a valid analysis directory. This is likely an error in the package.")

        # translate bar_order for R to read
        bar_order_str = str(bar_order).translate({ord(i): None for i in ' []'}) # remove spaces and brackets from bar_order

        # merge
        print("Running merge")
        from nett.analysis.merge import merge
        merge(run_dir, output_dir)

        print("Running analysis for [train]")
        from nett.analysis.train_viz import train_viz
        train_viz(output_dir, output_dir, ep_bucket, num_episodes)

        # test
        print("Running analysis for [test]")
        from nett.analysis.test_viz import test_viz
        test_viz(output_dir, chick_data_dir, output_dir, bar_order_str, color_bars)

        print(f"Analysis complete. See results at {output_dir}")

    @staticmethod
    def analyze(config: str,
                run_dir: str | Path,
                output_dir: Optional[str | Path] = None,
                ep_bucket: int = 100,
                num_episodes: int = 1000,
                bar_order: str | list[int] = "default",
                color_bars: bool = True) -> None:
        """
        Analyze the results of a run.

        This method is a static method and does not require an instance of the NETT class to be called.

        Args:
            config (str): The configuration of the experiment to be analyzed. It can be "parsing", "binding", "viewinvariant", "facedifferentiation", "biomotion", or "statisticallearning".
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

        analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
        if output_dir is None:
            output_dir = run_dir.joinpath("results")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        chick_data_dir = Path(analysis_dir).joinpath("ChickData", f"{config.lower()}.csv")

        if not chick_data_dir.exists():
            raise ValueError(f"'{config}' is not a valid config.")
        elif not run_dir.exists():
            raise ValueError(f"'{run_dir}' is not a valid run directory.")
        elif not analysis_dir.exists():
            raise ValueError(f"'{analysis_dir}' is not a valid analysis directory. This is likely an error in the package.")

        # translate bar_order for R to read
        bar_order_str = str(bar_order).translate({ord(i): None for i in ' []'}) # remove spaces and brackets from bar_order

        # merge
        print("Running merge")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_merge_csvs.R")),
                        "--logs-dir", str(run_dir),
                        "--results-dir", str(output_dir),
                        "--results-name", "analysis_data",
                        "--csv-train", "train_results.csv",
                        "--csv-test", "test_results.csv"], check=True)

        # train
        print("Running analysis for [train]")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_train_viz.R")),
                        "--data-loc", str(output_dir.joinpath("analysis_data")),
                        "--results-wd", str(output_dir),
                        "--ep-bucket", str(ep_bucket),
                        "--num-episodes", str(num_episodes)], check=True)

        # test
        print("Running analysis for [test]")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_test_viz.R")),
                        "--data-loc", str(output_dir.joinpath("analysis_data")),
                        "--results-wd", str(output_dir),
                        "--bar-order", bar_order_str,
                        "--color-bars", str(color_bars),
                        "--chick-file", str(chick_data_dir)], check=True)

        print(f"Analysis complete. See results at {output_dir}")

    def _execute_job(self, job: Job) -> Future:
        brain: "nett.Brain" = deepcopy(self.brain)
        
        if job.estimate_memory: # estimate memory uses train env for estimation
            modes = ["train"]
        elif job.mode == "full":
            modes = ["train", "test"]
        else: # test or train
            modes = [job.mode]

        # loop over modes to validate then run the environment
        for mode in modes:
            # validation run
            # self._run_env(
            #     mode=mode, 
            #     port=job.port, 
            #     kwargs = job.validation_kwargs(), 
            #     callback = check_env
            # )   
            # actual run
            self._run_env(
                mode=mode, 
                port=job.port, 
                kwargs = job.env_kwargs(), 
                callback = lambda env: getattr(brain, mode)(env, job) # grabs brain.train or brain.test based on mode
            )

        return f"Job Completed Successfully for Brain #{job.brain_id} with Condition: {job.condition}"

    def _estimate_job_memory(self, devices: list[int], base_port: int) -> int:
        self.logger.info("Estimating memory for a single job")
        try:
            # find the GPU with the most free memory
            free_memory = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device)).free for device in devices]
            most_free_gpu = free_memory.index(max(free_memory))

            # find unused port
            while port_in_use(base_port):
                base_port += 1

            # create a test job to estimate memory
            job = Job(
                brain_id=0, 
                condition=self.environment.imprinting_conditions[0], 
                device=most_free_gpu, 
                index=0,
                port=base_port,
                estimate_memory=True)

            job.save_checkpoints = False

            # change initial port for next job
            base_port += 1

            # calculate current memory usage for baseline for comparison
            pre_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(job.device)).used

            # initializer = mute if not verbose else None
            # executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
            executor = ProcessPoolExecutor(max_workers=1, initializer=None)
            job_sheet: dict[Future, dict[str, Job]] = {}

            # run job with estimate_memory set to True
            job_future = executor.submit(self._execute_job, job)
            job_sheet[job_future] = job

            future_wait(job_sheet, return_when=FIRST_COMPLETED)

            with open(Path.joinpath(job.paths["base"], "mem.txt").resolve(), "r") as file:
                post_memory = int(file.readline())
        except Exception as e:
            self.logger.exception(f"Error in estimating memory: {e}")
            raise e
        finally:
            if job.paths["base"].exists():
                shutil.rmtree(job.paths["base"])
        
        # estimate memory allocated
        return post_memory - pre_memory

    @staticmethod
    def _filter_job_sheet(job_sheet: dict[Future, dict[str,Any]], selected_columns: list[str]) -> list[dict[str,bool|str]]:
        # TODO include waitlisted jobs
        runStatus = lambda job_future: {'running': job_future.running()}
        jobInfo = lambda job: {k: getattr(job, k) for k in selected_columns}

        return [runStatus(job_future) | jobInfo(job) for job_future, job in job_sheet.items()]

    @staticmethod
    def _get_memory_status(devices: list[int]) -> dict[int, dict[str, int]]:
        unpack = lambda memory_status: {"free": memory_status.free, "used": memory_status.used, "total": memory_status.total}
        memory_status = {
            device_id : unpack(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))) 
            for device_id in devices
        }
        return memory_status
    
    def _launch_jobs(self, jobs: list[Job], wait: bool, waitlist: list[Job], verbose: bool) -> dict[Future, Job]:
        """
        Launch the jobs in the job sheet.

        Args:
            jobs (list[Job]): The jobs to be launched.
            waitlist (list[Job], optional): The jobs that are to be queued until memory is available.

        Returns:
            dict[Future, Job]: A dictionary of futures corresponding to the jobs that were launched from them.
        """
        try:
            max_workers = 1 if len(jobs) == 1 else os.cpu_count()
            initializer = mute if not verbose else None
            executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
            job_sheet: dict[Future, dict[str, Job]] = {}

            for job in jobs:
                job_future = executor.submit(self._execute_job, job)
                job_sheet[job_future] = job
                time.sleep(1)

            while waitlist:
                done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                for doneFuture in done:
                    doneJob: Job = job_sheet.pop(doneFuture)
                    freeDevice: int = doneJob.device
                    freePort: int = doneJob.port
                    job = waitlist.pop()
                    job.device = freeDevice
                    job.port = freePort
                    job_future = executor.submit(self._execute_job, job)
                    job_sheet[job_future] = job
                    time.sleep(1)

            if wait:
                while job_sheet:
                    done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                    for doneFuture in done:
                        job_sheet.pop(doneFuture)
                    time.sleep(1)

            # close processes and free up resources on completion
            executor.shutdown()

            return job_sheet
        except Exception as e:
            print(str(e))

    @staticmethod
    def _get_task_set(num_brains: int, all_conditions: list[str], conditions: Optional[list[str]]) -> set[tuple[str,int]]: #TODO: Create a better name for this method
        # create set of all conditions
        all_conditions_set: set[str] = set(all_conditions)
    
        # check if user-defined their own conditions
        if (conditions is not None):
            # create a set of user-defined conditions
            condition_set: set[str] = set(conditions)

            if not condition_set.issubset(all_conditions_set):
                raise ValueError(f"Unknown conditions: {conditions}. Available conditions are: {all_conditions}")
        # default to all conditions
        else:
            condition_set: set[str] = all_conditions_set

        # create set of all brain-environment combinations
        return set(product(condition_set, set(range(1, num_brains + 1))))

    def _schedule_jobs(self, task_set: set[tuple[str,int]], devices: list[int], job_memory: int, port: int, logger: "Logger") -> tuple[list[Job], list[Job]]:
        # create jobs
        jobs: list[Job] = []
        waitlist: list[Job] = []

        # assign devices based on memory availability
        # get the list of devices
        free_devices: list[int] = devices.copy()

        # get the free memory status for each device
        free_device_memory: dict[int, int] = {device: memory_status["free"] for device, memory_status in self._get_memory_status(devices).items()}

        while task_set:
            # if there are no free devices, add jobs to the waitlist
            if not free_devices:
                if not jobs:
                    raise ValueError("No jobs could be scheduled. Job size too large for GPUs. If job_memory='auto', consider setting buffer to 1. Otherwise, consider setting job_memory to a value less than or equal to total free GPU memory / buffer.")
                logger.info("No free devices. Jobs will be queued until a device is available.")
                waitlist = [
                    Job(brain_id, condition, device=-1, index=len(jobs)+i, port=-1) 
                    for i, (condition, brain_id) in enumerate(task_set)
                ]
                logger.warning("Insufficient GPU Memory. Jobs will be queued until memory is available. This may take a while.")
                break

            # remove devices that don't have enough memory
            if free_device_memory[free_devices[-1]] < job_memory:
                logger.info(f"Device {free_devices[-1]} does not have enough memory. Removing from list of available devices.")
                free_devices.pop()
            # assign device to job
            else:
                logger.info(f"Assigning device {free_devices[-1]} to job")
                # create job
                condition, brain_id = task_set.pop()

                # find unused port
                while port_in_use(port):
                    port += 1

                job = Job(
                    brain_id=brain_id, 
                    condition=condition, 
                    device=free_devices[-1], 
                    index=len(jobs),
                    port=port)
                jobs.append(job)

                # change initial port for next job
                port += 1

                # allocate memory
                free_device_memory[free_devices[-1]] -= job_memory
                # rotate devices
                free_devices = [free_devices[-1]] + free_devices[:-1]

        return jobs, waitlist

    @staticmethod
    def _validate_devices(devices: Optional[list[int]]) -> list[int]:
        # check if the devices are available and return the list of devices to be used
        available_devices: list[int] = list(range(nvmlDeviceGetCount()))

        if devices is None:
            devices = available_devices
        elif isinstance(devices, list) and not set(devices).issubset(set(available_devices)):
            raise ValueError("Custom device list lists unknown devices. Available devices are: {available_devices}")

        return devices

    def _run_env(self, mode: str, port: int, kwargs: dict[str,Any], callback):
        # run environment
        # can be train or test mode and can be for validation or actual run
        while True:
            try:
                # wrap environment
                # with self._wrap_env(mode, port, kwargs) as environment:
                # run the callback. This can be check_env or brain.train or brain.test
                if mode == "test":
                    def make_env(rank, seed=0):
                        def _init():
                            kwargs_copy = deepcopy(kwargs)
                            kwargs_copy["rank"] = rank
                            env_copy = self._wrap_env(mode, port+rank, kwargs_copy)
                            env_copy.reset(seed=seed + rank)
                            return env_copy

                        return _init

                    # initialize environment
                    num_envs = Job.test_eps
                    
                    envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])
                    callback(envs)
                    envs.close()
                else:
                    with self._wrap_env(mode, port, kwargs) as environment:
                        callback(environment)
                break
            # when running multiple runs in parallel, the port may be in use, so try the next port
            except UnityWorkerInUseException as _:
                self.logger.warning(f"Worker {port} is in use. Trying next port...")
                port += 1
            except Exception as ex:
                self.logger.exception(f"{mode} env validation failed: {str(ex)}" if kwargs["validation-mode"] \
                                      else f"{mode} env failed: {str(ex)}")  
                raise ex

    def _wrap_env(self, mode: str, port: int, kwargs: dict[str,Any]) -> "nett.Body":
        copy_environment = deepcopy(self.environment)
        copy_environment.initialize(mode, port, allow_multi_obs=self.body.binocular, **kwargs)
        copy_body = deepcopy(self.body)
        # apply wrappers (body)
        return copy_body(copy_environment)    