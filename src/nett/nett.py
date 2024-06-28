"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
   :synopsis: Main class for training, testing and analyzing brains in environments.

"""

import time
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy
from itertools import product, cycle
from concurrent.futures import ProcessPoolExecutor, Future, wait as future_wait, FIRST_COMPLETED

import pandas as pd
from sb3_contrib import RecurrentPPO
from pynvml import nvmlInitWithFlags, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from nett.utils.io import mute
from nett.utils.job import Job

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

    def __init__(self, brain: "nett.Brain", body: "nett.Body", environment: "nett.Environment") -> None:
        """
        Initialize the NETT class.
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.brain = brain
        self.body = body
        self.environment = environment
        # for NVIDIA memory management
        # flag 1 indicates that it will not throw an error if there is no NVIDIA GPU
        nvmlInitWithFlags(1)

    def run(self,
            output_dir: Path | str,
            num_brains: int = 1,
            mode: str = "full",
            train_eps: int = 1000,
            test_eps: int = 20,
            batch_mode: bool = True,
            device_type: str = "cuda",
            devices: list[int] | int =  -1,
            description: Optional[str] = None,
            job_memory: str | int = "auto",
            buffer: float = 1.2,
            steps_per_episode: int = 1000,
            conditions: Optional[list[str]] = None,
            verbosity: int = 1,
            run_id: str = '',
            synchronous=True,
            save_checkpoints: bool = False,
            checkpoint_freq: int = 30_000) -> list[Future]:
        """
        Run the training and testing of the brains in the environment.

        Args:
            output_dir (Path | str): The directory where the run results will be stored.
            num_brains (int, optional): The number of brains to be trained and tested. Defaults to 1.
            mode (str, optional): The mode in which the brains are to be trained and tested. It can be "train", "test", or "full". Defaults to "full".
            train_eps (int, optional): The number of episodes the brains are to be trained for. Defaults to 1000.
            test_eps (int, optional): The number of episodes the brains are to be tested for. Defaults to 20.
            batch_mode (bool, optional): Whether to run in batch mode, which will not display Unity windows. Good for headless servers. Defaults to True.
            device_type (str, optional): The type of device to be used for training and testing. It can only be "cuda" currently. Defaults to "cuda".
            devices (list[int] | int, optional): The list of devices to be used for training and testing. If -1, all available devices will be used. Defaults to -1.
            description (str, optional): A description of the run. Defaults to None.
            job_memory (int, optional): The memory allocated, in Gigabytes, for a single job. Defaults to 4.
            buffer (float, optional): The buffer for memory allocation. Defaults to 1.2.
            steps_per_episode (int, optional): The number of steps per episode. Defaults to 1000.
            verbosity (int, optional): The verbosity level of the run. Defaults to 1.
            run_id (str, optional): The run ID. Defaults to ''.
            save_checkpoints (bool, optional): Whether to save checkpoints during training. Defaults to False.
            checkpoint_freq (int, optional): The frequency at which checkpoints are saved. Defaults to 30_000.

        Returns:
            list[Future]: A list of futures representing the jobs that have been launched.

        Example:
            >>> job_sheet = benchmarks.run(output_dir="./test_run", num_brains=2, train_eps=100, test_eps=10) # benchmarks is an instance of NETT
        """
        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {self.output_dir.resolve()}")

        # register run config
        self.mode = mode
        self.verbosity = verbosity
        self.num_brains = num_brains
        self.train_eps = train_eps
        self.test_eps = test_eps
        self.description = description
        self.job_memory = job_memory
        self.buffer = buffer
        self.steps_per_episode = steps_per_episode
        self.device_type = self._validate_device_type(device_type)
        self.devices: list[int] | int = self._validate_devices(devices)
        self.batch_mode: bool = batch_mode
        self.run_id = run_id
        self.save_checkpoints = save_checkpoints
        self.checkpoint_freq = checkpoint_freq
        
        # schedule jobs
        jobs, waitlist = self._schedule_jobs(conditions=conditions)
        self.logger.info("Scheduled jobs")

        # launch jobs
        self.logger.info("Launching")
        job_sheet = self.launch_jobs(jobs, synchronous, waitlist)

        # return control back to the user after launching jobs, do not block
        return job_sheet

    def launch_jobs(self, jobs: list[Job], wait: bool, waitlist: list[Job] = [], ) -> dict[Future, Job]:
        """
        Launch the jobs in the job sheet.

        Args:
            jobs (list[Job]): The jobs to be launched.
            waitlist (list[Job], optional): The jobs that are to be queued until memory is available.

        Returns:
            dict[Future, Job]: A dictionary of futures corresponding to the jobs that were launched from them.
        """
        try:
            max_workers = 1 if len(jobs) == 1 else None
            initializer = mute if not self.verbosity else None
            executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
            job_sheet: dict[Future, dict[str, Job]] = {}

            for job in jobs:
                job_future = executor.submit(self._execute_job, job)
                job_sheet[job_future] = job
                time.sleep(1)

            while waitlist:
                done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                for doneFuture in done:
                    freeDevice: int = job_sheet.pop(doneFuture).device
                    job = waitlist.pop()
                    job.device = freeDevice
                    job_future = executor.submit(self._execute_job, job)
                    job_sheet[job_future] = job
                    time.sleep(1)

            if wait:
                while job_sheet:
                    done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                    for doneFuture in done:
                        job_sheet.pop(doneFuture)
                    time.sleep(1)

            return job_sheet
        except Exception as e:
            print(str(e))

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

# TODO v0.3, make .analyze() a staticmethod so that it does not need a class instance to call
    # TODO v0.3. add support for user specified output_dir
    # Discussion v0.3 is print okay or should we have it log using nett's logger?
    # Discussion v0.3 move this out of the class entirely? from nett import analyze, analyze(...)

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
            config (str): The configuration of the experiment to be analyzed. It can be "parsing", "binding", or "viewinvariant".
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
        analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
        if output_dir is None:
            output_dir = run_dir.joinpath("results")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        chick_data_dir = Path(analysis_dir).joinpath("ChickData", f"{config.lower()}.csv")

        if not chick_data_dir.exists():
            raise ValueError(f"'{config}' is not a valid config.")

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

    def _schedule_jobs(self, conditions: Optional[list[str]] = None) -> tuple[list[Job], list[Job]]:
        # create jobs
        
        # create set of all conditions
        all_conditions: set[str] = set(self.environment.config.conditions)
    
        # check if user-defined their own conditions
        if (conditions is not None):
            # create a set of user-defined conditions
            user_conditions: set[str] = set(conditions)

            if user_conditions.issubset(all_conditions):
                self.logger.info(f"Using user specified conditions: {conditions}")
                # create set of all brain-environment combinations for user-defined conditions
                task_set: set[tuple[str,int]] = set(product(user_conditions, set(range(1, self.num_brains + 1))))
            else:
                raise ValueError(f"Unknown conditions: {conditions}. Available conditions are: {self.environment.config.conditions}")
        # default to all conditions
        else:
            # create set of all brain-environment combinations
            task_set: set[tuple[str,int]] = set(product(all_conditions, set(range(1, self.num_brains + 1))))

        jobs: list[Job] = []
        waitlist: list[Job] = []

        # assign devices based on memory availability
        # get the list of devices
        free_devices: list[int] | int = self.devices.copy()

        # get the free memory status for each device
        free_device_memory: dict[int, int] = {device: memory_status["free"] for device, memory_status in self._get_memory_status().items()}

        #TODO: maybe waitlist all processes and then run them once a the initial job is complete, try running that for a short period of time?

        # estimate memory for a single job
        job_memory: float = self.buffer * self._estimate_job_memory(free_device_memory)
        print('JOB MEMORY:: ', job_memory)

        exit()

        while task_set:
            # if there are no free devices, add jobs to the waitlist
            if not free_devices:
                waitlist = [
                    Job(brain_id, condition, -1, self.output_dir, len(jobs)+i) 
                    for i, (condition, brain_id) in enumerate(task_set)
                ]
                self.logger.warning("Insufficient GPU Memory. Jobs will be queued until memory is available. This may take a while.")
                break
            # remove devices that don't have enough memory
            elif free_device_memory[free_devices[-1]] < job_memory:
                free_devices.pop()
            # assign device to job
            else:
                # allocate memory
                free_device_memory[free_devices[-1]] -= job_memory
                # create job
                condition, brain_id = task_set.pop()
                job = Job(brain_id, condition, free_devices[-1], self.output_dir, len(jobs))
                jobs.append(job)

        return jobs, waitlist

    def _wrap_env(self, mode: str, kwargs: dict[str,Any]) -> "nett.Body":
        copy_environment = deepcopy(self.environment)
        copy_environment.initialize(mode=mode, **kwargs)
        # apply wrappers (body)
        return self.body(copy_environment)

    def _execute_job(self, job: Job) -> Future:

        # for train
        if self.mode not in ["train", "test", "full"]:
            raise ValueError(f"Unknown mode type {self.mode}, should be one of ['train', 'test', 'full']")

        brain: "nett.Brain" = deepcopy(self.brain)

        # common environment kwargs
        kwargs = {"rewarded": bool(brain.reward),
                  "rec_path": str(job.paths["env_recs"]),
                  "log_path": str(job.paths["env_logs"]),
                  "condition": job.condition,
                  "run_id": job.brain_id,
                  "episode_steps": self.steps_per_episode,
                  "device_type": self.device_type,
                  "batch_mode": self.batch_mode}

        # for train
        if self.mode in ["train", "full"]:
            try:
                # initialize environment with necessary arguments
                train_environment = self._wrap_env("train", kwargs)
                # calculate iterations
                iterations = self.steps_per_episode * self.train_eps
                # train
                brain.train(
                    env=train_environment,
                    iterations=iterations,
                    device_type=self.device_type,
                    device=job.device,
                    index=job.index,
                    paths=job.paths,
                    save_checkpoints=self.save_checkpoints,
                    checkpoint_freq=self.checkpoint_freq,)
                train_environment.close()
            except Exception as e:
                self.logger.error(f"Error in training: {e}")
                train_environment.close()
                exit()    

        # for test
        if self.mode in ["test", "full"]:
            try:
                # initialize environment with necessary arguments
                test_environment = self._wrap_env("test", kwargs)
                # calculate iterations
                iterations = self.test_eps * test_environment.config.num_conditions

                # Q: Why in test but not in train?
                if not issubclass(brain.algorithm, RecurrentPPO):
                    iterations *= self.steps_per_episode

                # test
                brain.test(
                    env=test_environment,
                    iterations=iterations,
                    model_path=str(job.paths['model'].joinpath('latest_model.zip')),
                    rec_path = str(job.paths["env_recs"]),
                    index=job.index)
                test_environment.close()
            except Exception as e:
                self.logger.error(f"Error in testing: {e}")
                test_environment.close()
                exit()

        return f"Job Completed Successfully for Brain #{job.brain_id} with Condition: {job.condition}"

    # pylint: disable-next=unused-argument
    def _estimate_job_memory(self, device_memory_status: dict) -> int: # pylint: disable=unused-argument
        # TODO (v0.5) add a dummy job to gauge memory consumption

        # # get device with the maxmium memory available
        # max_memory_device = max(device_memory_status,
        #                         key=lambda device: device_memory_status[device].free)
        # # send a dummy job to gauge memory consumption
        # dummy_job = self._create_job(device=max_memory_device,
        #                              mode=random.choice(self.environment.config.modes),
        #                              brain_id=-1)

        # # send the job
        # self.logger.info("Estimating memory by executing a single job")

        # return a hurestic value for now (4GiB per job)
        # multiply to return in bytes
        if (self.job_memory == "auto"):
            try:
                if (self.job_memory == "auto"):
                    currentMemory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).used
                    tmp_path = self.output_dir / ".tmp/"
                    tmp_path.mkdir(parents=True, exist_ok=True)
                    job = Job(0, self.environment.config.conditions[0], 0, tmp_path, 0) #TODO: May need to change the card being chosen here

                    brain: "nett.Brain" = deepcopy(self.brain)

                    # common environment kwargs
                    kwargs = {"rewarded": bool(brain.reward),
                            "rec_path": str(job.paths["env_recs"]),
                            "log_path": str(job.paths["env_logs"]),
                            "condition": job.condition,
                            "run_id": job.brain_id,
                            "episode_steps": self.steps_per_episode,
                            "device_type": self.device_type,
                            "batch_mode": self.batch_mode}

                    # for train
                    if self.mode in ["train", "full"]:
                        try:
                            # initialize environment with necessary arguments
                            train_environment = self._wrap_env("train", kwargs)
                            # calculate iterations
                            iterations = self.steps_per_episode * self.train_eps
                            # train
                            brain.testrun(
                                env=train_environment,
                                iterations=iterations,
                                device_type=self.device_type,
                                device=job.device,
                                index=job.index,
                                paths=job.paths,
                                save_checkpoints=False,
                                checkpoint_freq=self.checkpoint_freq,)
                            train_environment.close()
                        except Exception as e:
                            self.logger.error(f"Error in training: {e}")
                            train_environment.close()
                            exit() 
                    else:
                        raise NotImplementedError("Only train or full mode is supported for now") 

                    with open("./.tmp/memory_use", "r") as file:
                        memory_allocated = int(file.readline()) - currentMemory
                    self.logger.info(f"Estimated memory: {memory_allocated}")
            except Exception as e:
                self.logger.error(f"Error in estimating memory: {e}", exc_info=1)
                exit()
            finally:
                if (self.output_dir / ".tmp/").exists():
                    shutil.rmtree(self.output_dir / ".tmp/")
        else:
            memory_allocated = self.job_memory * (1024 * 1024 * 1024)
        return memory_allocated

    def _filter_job_sheet(self, job_sheet: dict[Future, dict[str,Any]], selected_columns: list[str]) -> list[dict[str,bool|str]]:
        # TODO include waitlisted jobs
        runStatus = lambda job_future: {'running': job_future.running()}
        jobInfo = lambda job: {k: getattr(job, k) for k in selected_columns}

        return [runStatus(job_future) | jobInfo(job) for job_future, job in job_sheet.items()]

    def _get_memory_status(self) -> dict[int, dict[str, int]]:
        unpack = lambda memory_status: {"free": memory_status.free, "used": memory_status.used, "total": memory_status.total}
        memory_status = {
            device_id : unpack(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))) 
            for device_id in self.devices
        }
        return memory_status

    def _validate_device_type(self, device_type: str) -> str:
        # TODO (v0.5) add automatic type checking usimg pydantic or similar
        if device_type not in ["cuda"]:
            raise ValueError("Should be one of ['cuda']")

        return device_type

    def _validate_devices(self, devices: list[int] | int) -> list[int]:
        # check if the devices are available and return the list of devices to be used
        available_devices: list[int] = list(range(nvmlDeviceGetCount()))

        if devices == -1:
            devices = available_devices
        elif isinstance(devices, list) and not set(devices).issubset(set(available_devices)):
            raise ValueError("Custom device list lists unknown devices. Available devices are: {available_devices}")

        self.logger.info(f"Devices that will be used: {devices}")

        return devices

    def summary(self) -> None: # TODO: only raises a NotImplementedError for now
        '''Generate a toml file and save it to the run directory.'''
        raise NotImplementedError
