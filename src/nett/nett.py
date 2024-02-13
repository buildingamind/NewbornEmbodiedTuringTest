"""
This module contains the NETT class, which is the main class for training and testing brains in environments.

.. module:: nett
   :synopsis: Main class for training and testing brains in environments.

"""

import time
import subprocess
from pathlib import Path
from typing import Any
from copy import deepcopy
from itertools import product
from concurrent.futures import ProcessPoolExecutor, Future

import pandas as pd
from sb3_contrib import RecurrentPPO
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from nett import Brain, Body, Environment
from nett.utils.io import mute
# from brain.builder import Brain
# from body.builder import Body
# from environment.builder import Environment
# from utils.io import mute

class NETT:
    """
    The NETT class is the main class for training and testing brains in environments.

    :param brain: The brain to be trained and tested.
    :type brain: Brain
    :param body: The body to be used for training and testing the brain.
    :type body: Body
    :param environment: The environment in which the brain is to be trained and tested.
    :type environment: Environment
    
    :ivar output_dir: The directory where the run results will be stored.
    :vartype output_dir: Path
    :ivar mode: The mode in which the brains are to be trained and tested. It can be "train", "test", or "full".
    :vartype mode: str
    :ivar verbosity: The verbosity level of the run.
    :vartype verbosity: int
    :ivar num_brains: The number of brains to be trained and tested.
    :vartype num_brains: int
    :ivar train_eps: The number of episodes the brains are to be trained for.
    :vartype train_eps: int
    :ivar test_eps: The number of episodes the brains are to be tested for.
    :vartype test_eps: int
    :ivar description: A description of the run.
    :vartype description: str
    :ivar buffer: The buffer for memory allocation.
    :vartype buffer: float
    :ivar step_per_episode: The number of steps per episode.
    :vartype step_per_episode: int
    :ivar device_type: The type of device to be used for training and testing. It can be "cuda" or "cpu".
    :vartype device_type: str
    :ivar devices: The list of devices to be used for training and testing. If -1, all available devices will be used.
    :vartype devices: list[int] | int
    :ivar brain: The brain to be trained and tested.
    :vartype brain: Brain
    :ivar body: The body to be used for training and testing the brain.
    :vartype body: Body
    :ivar environment: The environment in which the brain is to be trained and tested.
    :vartype environment: Environment
    :ivar logger: The logger for the NETT class.
    :vartype logger: Logger

    Example:

    >>> from nett import NETT
    >>> # create a brain, body, and environment
    >>> benchmarks = NETT(brain, body, environment)
    """

    def __init__(self, brain: Brain, body: Body, environment: Environment) -> None:
        """
        Initialize the NETT class.

        :param brain: The brain to be trained and tested.
        :type brain: Brain
        :param body: The body to be used for training and testing the brain.
        :type body: Body
        :param environment: The environment in which the brain is to be trained and tested.
        :type environment: Environment
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.brain = brain
        self.body = body
        self.environment = environment
        # for NVIDIA memory management
        nvmlInit()

    def run(self,
            output_dir: Path | str,
            num_brains: int = 1,
            mode: str = "full",
            train_eps: int = 1000,
            test_eps: int = 20,
            device_type: str = "cuda",
            devices: list[int] | int =  -1,
            description: str = None,
            buffer: float = 1.2,
            step_per_episode: int = 200,
            verbosity: int = 0) -> list[Future]: # pylint: disable=unused-argument
        """
        Run the training and testing of the brains in the environment.

        :param output_dir: The directory where the run results will be stored.
        :type output_dir: Path | str
        :param num_brains: The number of brains to be trained and tested.
        :type num_brains: int, optional
        :param mode: The mode in which the brains are to be trained and tested. It can be "train", "test", or "full".
        :type mode: str, optional
        :param train_eps: The number of episodes the brains are to be trained for.
        :type train_eps: int, optional
        :param test_eps: The number of episodes the brains are to be tested for.
        :type test_eps: int, optional
        :param device_type: The type of device to be used for training and testing. It can be "cuda" or "cpu".
        :type device_type: str, optional
        :param devices: The list of devices to be used for training and testing. If -1, all available devices will be used.
        :type devices: list[int] | int, optional
        :param description: A description of the run.
        :type description: str, optional
        :param buffer: The buffer for memory allocation.
        :type buffer: float, optional
        :param step_per_episode: The number of steps per episode.
        :type step_per_episode: int, optional
        :param verbosity: The verbosity level of the run.
        :type verbosity: int, optional
        
        :return: A list of futures representing the jobs that have been launched.
        :rtype: list[Future]

        Example:
    
        >>> job_sheet = benchmarks.run(output_dir="./test_run", num_brains=2, train_eps=100, test_eps=10) # benchmarks is an instance of NETT
        """
        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {self.output_dir.resolve()}")

        # TODO (v0.3) upgrade to toml format
        # register run config
        self.mode = mode
        self.verbosity = verbosity
        self.num_brains = num_brains
        self.train_eps = train_eps
        self.test_eps = test_eps
        self.description = description
        self.buffer = buffer
        self.step_per_episode = step_per_episode
        self.device_type = self._validate_device_type(device_type)
        self.devices = self._validate_devices(devices)

        # schedule jobs
        jobs = self._schedule_jobs()
        self.logger.info("Scheduled jobs")

        # launch jobs
        self.logger.info("Launching")
        job_sheet = self.launch_jobs(jobs)

        # return control back to the user after launching jobs, do not block
        return job_sheet

    def launch_jobs(self, jobs: list[dict]) -> list[Future]:
        """
        Launch the jobs in the job sheet.

        :param jobs: The jobs to be launched.
        :type jobs: list[dict]
        
        :return: A list of futures representing the jobs that have been launched.
        :rtype: list[Future]
        """
        max_workers = 1 if len(jobs) == 1 else None
        initializer = mute if not self.verbosity else None
        executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
        job_sheet = []
        for job in jobs:
            job_future = executor.submit(self._execute_job, job)
            job_sheet.append({"running": job_future, "specification": job})
            time.sleep(1) # environment creation sometimes fails if no pause is given between consecutive job submissions
# with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer) as executor:
        #     future_to_job = {}
        #     for job in jobs:
        #         future_to_job[executor.submit(self._execute_job, job)] = job
        #         time.sleep(1)
        #     for future in as_completed(future_to_job):
        #         job = future_to_job[future]
        #         try:
        #             data = future.result()
        #         except Exception as exc:
        #             print("%r generated an exception: %s" % (job["brain_id"], exc))
        return job_sheet

    def status(self, job_sheet: dict[Future, dict]) -> pd.DataFrame:
        """
        Get the status of the jobs in the job sheet.
        
        :param job_sheet: The job sheet returned by the .launch_jobs() method.
        :type job_sheet: dict[Future, dict]
            
        :return: A dataframe containing the status of the jobs in the job sheet.
        :rtype: pd.DataFrame

        Example:

        >>> status = benchmarks.status(job_sheet) # benchmarks is an instance of NETT, job_sheet is the job sheet returned by the .run() method
        """
        selected_columns = ["brain_id", "condition", "device"]
        filtered_job_sheet = [self._filter_job_record(job_record, selected_columns) for job_record in job_sheet]
        return pd.json_normalize(filtered_job_sheet)

# TODO v0.3, make .analyze() a staticmethod so that it does not need a class instance to call
    # TODO v0.3. add support for user specified output_dir
    # Discussion v0.3 is print okay or should we have it log using nett's logger?
    # Discussion v0.3 move this out of the class entirely? from nett import analyze, analyze(...)

    @staticmethod
    def analyze(run_dir: str | Path,
                output_dir: str | Path | None = None,
                ep_bucket: int = 100,
                num_episodes: int = 1000) -> None:
        """
        Analyze the results of a run. This method is a static method and does not require an instance of the NETT class to be called.
        
        :param run_dir: The directory where the run results are stored.
        :type run_dir: str | Path
        :param output_dir: The directory where the analysis results will be stored. If None, the analysis results will be stored in the run directory.
        :type output_dir: str | Path | None, optional
        :param ep_bucket: The number of episodes to be grouped together for analysis.
        :type ep_bucket: int, optional
        :param num_episodes: The number of episodes to be analyzed.
        :type num_episodes: int, optional
            
        :return: None
        :rtype: None

        Example:

        >>> benchmarks.analyze(run_dir="./test_run", output_dir="./results") # benchmarks is an instance of NETT
        """
        # set paths
        run_dir = Path(run_dir).resolve()
        analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
        if output_dir is not None:
            output_dir = Path(output_dir).resolve()
        else:
            output_dir = run_dir.joinpath("results")
        output_dir.mkdir(exist_ok=True)
        print(run_dir)
        print(analysis_dir)
        print(output_dir)

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
                        "--key-csv", str(analysis_dir.joinpath("Keys", "segmentation_key_new.csv")),
                        "--color-bars", "true",
                        "--chick-file", str(analysis_dir.joinpath("ChickData", "ChickData_Parsing.csv"))], check=True)

        print(f"Analysis complete. See results at {output_dir}")

    def _schedule_jobs(self):
        # get the free memory status for each device in list
        free_device_memory = {device: memory_status["free"] for device, memory_status in self._get_memory_status().items()}

        # estimate memory for a single job
        job_memory = self._estimate_job_memory(free_device_memory)

        # create jobs
        # create list of all brain-environment combinations
        # TODO (v0.5) replace environment.config.conditions and task_list with np.arrays to improve performance
        task_list = list(product(self.environment.config.conditions, list(range(1, self.num_brains + 1))))

        # assign devices based on memory to a condition and brain combination
        # TODO (v0.5) what's better than a loop / brute force here?
        jobs = []

        free_devices = list(free_device_memory.keys()) # list of device numbers of free devices
        dev_num = 0 # current device number

        for (condition, brain_id) in task_list:
            # find a device that has enough memory
            # FIXME self.buffer does not seem like a reliable metric for estimating memory consumption (by multiplying 1.2 to job_memory)
            while free_device_memory[free_devices[dev_num]] < job_memory * self.buffer:
                dev_num += 1
                # TODO (v0.3) allow partial execution
                if dev_num >= len(free_device_memory):
                    raise RuntimeError("Insufficient GPU Memory, could not create jobs for all tasks")
            # allocate memory
            free_device_memory[free_devices[dev_num]] -= job_memory*self.buffer
            # create job
            job = self._create_job(brain_id=brain_id, condition=condition, device=free_devices[dev_num])
            jobs.append(job)

        return jobs

    def _create_job(self, device: int, condition: str, brain_id: int) -> dict:
        # creates brain, env copies for a job
        brain_copy = deepcopy(self.brain)

        # configure paths for the job
        paths = self._configure_job_paths(condition=condition, brain_id=brain_id)

        # create job
        return {"brain": brain_copy,
                "environment": self.environment,
                "body": self.body,
                "device": device,
                "condition": condition,
                "brain_id": brain_id,
                "paths": paths}

    def _configure_job_paths(self, condition: str, brain_id: int) -> dict:
        subdirs = ["model", "checkpoints", "plots", "logs", "env_recs", "env_logs"]
        job_dir = Path.joinpath(self.output_dir, condition, f"brain_{brain_id}")
        paths = {subdir: Path.joinpath(job_dir, subdir) for subdir in subdirs}
        return paths

    def _execute_job(self, job: dict[str, Any]) -> Future:
        # common environment kwargs
        kwargs = {"rewarded": bool(self.brain.reward),
                  "rec_path": str(job["paths"]["env_recs"]),
                  "log_path": str(job["paths"]["env_logs"]),
                  "condition": str(job["condition"]),
                  "run_id": str(job["brain_id"])}

        # for train
        if self.mode in ["train", "full"]:
            # initialize environment with necessary arguments
            train_environment = deepcopy(job["environment"])
            train_environment.initialize(mode="train", **kwargs)
            # apply wrappers (body)
            train_environment = job["body"](train_environment)
            # train
            job["brain"].train(env=train_environment,
                               iterations=self.step_per_episode * self.train_eps,
                               device_type=self.device_type,
                               device=job["device"],
                               paths=job["paths"])
            return "Job completed successfully"

        # for test
        if self.mode in ["test", "full"]:
            # initialize environment with necessary arguments
            test_environment = deepcopy(job["environment"])
            test_environment.initialize(mode="test", **kwargs)
            # apply wrappers (body)
            test_environment = job["body"](test_environment)
            # test
            # readability over symmetry, sadly :(
            if issubclass(job["brain"].algorithm, RecurrentPPO):
                iterations = self.test_eps * test_environment.config.num_conditions
            else:
                iterations = self.step_per_episode * self.test_eps * job["environment"].config.num_conditions
            job["brain"].test(env=test_environment,
                              iterations=iterations,
                              model_path=f"{job['paths']['model'].joinpath('latest_model.zip')}")

        if self.mode not in ["train", "test", "full"]:
            raise ValueError(f"Unknown mode type {self.mode}, should be one of ['train', 'test', 'full']")

        return "Job Successfully Concluded"

    def _get_memory_status(self) -> dict[int, dict[str, int]]:
        unpack = lambda memory_status: {"free": memory_status.free, "used": memory_status.used, "total": memory_status.total}
        memory_status = {device_id : unpack(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id)))
                         for device_id in self.devices}
        return memory_status

    # pylint: disable-next=unused-argument
    def _estimate_job_memory(self, device_memory_status: dict) -> int: # pylint: disable=unused-argument
        # TODO (v0.3) add a dummy job to gauge memory consumption

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
        memory_allocated = 4 * (1024 * 1024 * 1024)
        return memory_allocated

    def _filter_job_record(self, job_record: dict, selected_columns: list[str]) -> dict:
        filter_specification = lambda specification: {key: value for key, value in specification.items() if key in selected_columns}
        return {key: (filter_specification(value) if isinstance(value, dict) else value.running()) for key, value in job_record.items()}

    def _validate_device_type(self, device_type: str):
        # TODO (v0.4) add automatic type checking usimg pydantic or similar
        if device_type in ["cuda", "cpu"]:
            pass
        else:
            raise ValueError("Should be one of ['gpu', 'cpu']")
        return device_type

    def _validate_devices(self, devices: list[int] | int):
        if self.device_type == "cpu" and isinstance(devices, list):
            raise ValueError("Custom device lists not supported for 'cpu' device")

        available_devices = list(range(nvmlDeviceGetCount()))
        if isinstance(devices, list) and len(devices) > nvmlDeviceGetCount():
            raise ValueError("Custom device lists not supported for 'cpu' device")

        if devices == -1:
            devices = available_devices
            self.logger.info(f"Devices that will be used: {devices}")
        elif isinstance(devices, list):
            for device in devices:
                if not isinstance(device, int) or device not in available_devices:
                    raise ValueError(f"Device [{device}] is invalid. Available devices are: {available_devices}")
            self.logger.info(f"Devices that will be used: {devices}")
        else:
            pass
        return devices

    def summary(self): # TODO: only raises a NotImplementedError for now
        '''Generate a toml file and save it to the run directory.'''
        raise NotImplementedError
    