import time
import pandas as pd

from nett import Brain, Body, Environment
from pathlib import Path
from typing import Any
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from sb3_contrib import RecurrentPPO
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


class NETT:
    def __init__(self, brain: Brain, body: Body, environment: Environment) -> None:
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.brain = brain
        self.body = body
        self.environment = environment     
        # for NVIDIA memory management   
        nvmlInit()

    def run(self, 
            dir: Path | str,
            num_brains: int = 1, 
            mode: str = "full", 
            train_eps: int = 1000, 
            test_eps: int = 20, 
            device_type: str = "cuda",
            devices: list[int] | int =  -1, 
            description: str = None, 
            buffer: float = 0.2,
            step_per_episode: int = 200,
            memory_per_brain: float = 0.5):
        # set up the run_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        self.run_dir = Path(dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {self.run_dir.resolve()}")

        # TO DO (v0.3) upgrade to toml format
        # register run config
        self.mode = mode
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
        self.logger.info(f"Scheduled jobs")

        # launch jobs
        self.logger.info(f"Launching")
        job_sheet = self.launch_jobs(jobs)
        
        # return control back to the user after launching jobs, do not block
        return job_sheet
    
    def launch_jobs(self, jobs: list[dict]) -> list[Future]:
        max_workers = 1 if len(jobs) == 1 else None
        executor = ProcessPoolExecutor(max_workers=max_workers)
        job_sheet = []
        for job in jobs:
            job_future = executor.submit(self._execute_job, job)
            job_sheet.append({'running': job_future, 'specification': job})
            time.sleep(1) # environment creation sometimes fails if no pause is given between consecutive job submissions
        return job_sheet
    
    def status(self, job_sheet: dict[Future, dict]) -> pd.DataFrame:
        selected_columns = ['brain_id', 'mode', 'device']
        filtered_job_sheet = [self._filter_job_record(job_record, selected_columns) for job_record in job_sheet]
        return pd.json_normalize(filtered_job_sheet)

    def analyze(self):
        
        raise NotImplementedError

    def _schedule_jobs(self):
        # get the free memory status for each device in list
        free_device_memory = {device: memory_status['free'] for device, memory_status in self._get_memory_status().items()}

        # estimate memory for a single job
        job_memory = self._estimate_job_memory(free_device_memory)

        # create jobs
        task_list = list(product(self.environment.config.modes, list(range(1, self.num_brains + 1))))
        # assign devices based on memory to a mode and brain combination
        # TO DO (v0.5) what's better than a loop / brute force here? 
        # TO DO (v0.3) replace explicit loop with .map() if does not sacrifice readability
        jobs = []
        # brute force iterate through devices
        for device, free_memory in free_device_memory.items():
            # iterate through modes and brain_ids
            for (mode, brain_id) in task_list:
                # check if memory is greater than job memory
                if free_memory > job_memory * self.buffer:
                    # alot device to task to make it a job which will be executed
                    job = self._create_job(brain_id=brain_id, mode=mode, device=device)
                    jobs.append(job)
                    # remove task from list
                    task_list.remove((mode, brain_id))
                    # update free memory
                    free_device_memory[device] = free_memory - job_memory
                else:
                    break
        # ensure all tasks have been converted to jobs
        # TO DO (v0.3) allow partial execution
        if task_list:
            raise RuntimeError("Insufficient GPU Memory, could not create jobs for all tasks")
        return jobs
    
    def _create_job(self, device: int, mode: str, brain_id: int) -> dict:
        # creates brain, env copies for a job
        brain_copy = deepcopy(self.brain)

        # configure paths for the job 
        paths = self._configure_job_paths(mode=mode, brain_id=brain_id)

        # create job
        return {'brain': brain_copy, 
                'environment': self.environment, 
                'body': self.body, 
                'device': device, 
                'mode': mode,
                'brain_id': brain_id,
                'paths': paths}
    
    def _configure_job_paths(self, mode: str, brain_id: int) -> dict:
        subdirs = ['model', 'checkpoints', 'plots', 'logs', 'env_recs', 'env_logs']
        job_dir = Path.joinpath(self.run_dir, mode, f'brain_{brain_id}')
        paths = {subdir: Path.joinpath(job_dir, subdir) for subdir in subdirs}
        return paths

    def _execute_job(self, job: dict[str, Any]) -> Future:
        # common environment kwargs
        kwargs = {'rewarded': True if self.brain.reward else False, 
                  'rec_path': str(job['paths']['env_recs']), 
                  'log_path': str(job['paths']['env_logs']), 
                  'mode': str(job['mode']),
                  'run_id': str(job['brain_id'])}
        
        # for train
        if self.mode in ["train", "full"]:
            # initialize environment with necessary arguments
            train_environment = deepcopy(job['environment'])
            train_environment.initialize(mode="train", **kwargs)
            # apply wrappers (body)
            train_environment = job['body'](train_environment)
            # train
            job['brain'].train(env=train_environment, 
                               iterations=self.step_per_episode * self.train_eps, 
                               device_type=self.device_type, 
                               device=job['device'], 
                               paths=job['paths'])

        # for test
        if self.mode in ["test", "full"]:
            # initialize environment with necessary arguments
            test_environment = deepcopy(job['environment'])
            test_environment.initialize(mode="test", **kwargs)
            # apply wrappers (body)
            test_environment = job['body'](test_environment)
            # test
            # readability over symmetry, sadly :(
            if issubclass(job['brain'].algorithm, RecurrentPPO):
                iterations = self.test_eps * test_environment.config.num_conditions
            else:
                iterations = self.step_per_episode * self.test_eps * job['environment'].config.num_conditions
            job['brain'].test(env=test_environment, 
                              iterations=iterations, 
                              model_path=f"{job['paths']['model'].joinpath('latest_model.zip')}")
            
        if self.mode not in ["train", "test", "full"]:
            raise ValueError(f"Unknown mode type {self.mode}, should be one of ['train', 'test', 'full']")
        
        return "Job Successfully Concluded"
    
    def _get_memory_status(self) -> dict[int, dict[str, int]]:
        unpack = lambda memory_status: {'free': memory_status.free, 'used': memory_status.used, 'total': memory_status.total}
        memory_status = {device_id : unpack(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))) 
                         for device_id in self.devices}
        return memory_status

    def _estimate_job_memory(self, device_memory_status: dict) -> int:
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
        selected_columns = ['brain_id', 'mode', 'device']
        filter_specification = lambda specification: {key: value for key, value in specification.items() if key in selected_columns}
        return {key: (filter_specification(value) if isinstance(value, dict) else value.running()) for key, value in job_record.items()}
    
    def _validate_device_type(self, device_type: str):
        # TO DO (v0.4) add automatic type checking usimg pydantic or similar
        if device_type in ["cuda", "cpu"]:
            pass
        else:
            raise ValueError("Should be one of ['gpu', 'cpu']")
        return device_type

    def _validate_devices(self, devices: list[int] | int):
        if self.device_type == "cpu" and isinstance(devices, list):
            raise ValueError("Custom device lists not supported for 'cpu' device")
        else:
            available_devices = list(range(nvmlDeviceGetCount()))
            if isinstance(devices, list) and len(devices) > nvmlDeviceGetCount():
                raise ValueError("Custom device lists not supported for 'cpu' device")
            elif devices == -1:
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

    # generate toml file and save to run directory
    def summary():
        raise NotImplementedError