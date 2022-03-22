# Embodied Pipeline Simulations
This folder contains all the necessary components and code for running simulations
from the embodied pipeline. The run python script provides
a range of parameters for each simulation.


Prior to using the script in this repository users should
have generated at least one executable as describe in the 
main README. 

Each user should try running a simulation locally before
trying on the server since it will be easier to notice something going wrong.


## Setting Up A Virtual Environment
Why use a virtual environment? Virtual environments help keep dependencies required by different projects separate.
A virtual environment will serve as a container to all the necessary python packages for this pipeline.
1. Install virtualenv

MAC / LINUX:
```
python3 -m pip install --user virtualenv
```
WINDOWS:
```
py -m pip install --user virtualenv
```
2. Create a virtual environment. The virts directory is designed to house your virtual environment. It will be ignored by git which means that when creating your first virtual environment you will explicitly have to create it. 

MAC / LINUX:
```
mkdir virts
cd virts
python3 -m venv VENV
```
WINDOWS:
```
mkdir virts
cd virts
py -m venv VENV
```
3. Activate your virtual environment. This step will need to be done every time a new terminal is used. Although, previously created virtual environments will persist as directories.

MAC / LINUX:
```
source VENV/bin/activate
```
WINDOWS:
```
.\VENV\Scripts\activate
```
4. Install the necessary directories. This step must be after the virtual environment is activated otherwise permission errors will appear. To simplify this install a `requirements.txt` is included.

MAC / LINUX / WINDOWS:
```
python3 -m pip install -r ./requirements.txt
```
5. Your virtual environment should be activated whenever you want to train and test models with this version of the environment. However, when you are finished with ml-agents, you can deactivate the virtual environment:
```
deactivate
```

## Running Experiments

#### Part 0: Compile Unity Environment
Follow instructions in the main directory to compile the unity executables for the environments.

#### Part 1: Run Experiments
Experiments consist of 2 phases. In the first phase is a 
training phase where the model is simulated in an environment
only consisting of the desired imprinting stimulus. The 
second phase is the testing phase. In this phase the agent
is tested for their recognition abilities across a range of
distractor stimuli (should I add a picture here?).

### Training
1. Prepare a [trainer configuration file](https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Training-Configuration-File.md) and adjust hyperparameters in the config file. Most will remain as is, in particular, filenames and directory locations should remain untouched unless necessary.

2. Run `python run.py` command. Set the run_id which you will
need for analysis later.
```
python run.py run_id=AgentBarney
```
* You can configure the environment by passing options. For example,  `env_args.episode-steps=2000` or `seed=1`. Below is the list of `--env-arg` options that are relevant for training:
```
--input_resolution INPUT_RESOLUTION
                      Resolution of visual inputs to agent. (default: 64)
--episode_steps EPISODE_STEPS
                      Number of steps per episode. (default: 1000)
--imprint_video VIDEO_PATH
                      ABSOLUTE path to the video containing imprinting object. If you specify a relative path (e.g., /my_video_dir), nothing will play on the display walls. You need to use an absolute path (e.g., /home/user/research/chickAI/my_video_dir). This video will be played on the right monitor first. The video needs to be a webm file if you are using Linux/Unix. (default: None)
--log_dir LOG_DIR     Directory to save environment logs. You NEED to provide a log-dir. Seriously. You won't get a log of where the agent is without it. (default: None)
--record_agent
                      Record the agent's camera if true. (default: False) Unlike most of the other flags, you do not need to provide a value after this flag. If you do not include the flag, it will be false. If you do include the flag, it will be true.        
--record_chamber
                      Record the chamber camera if true. (default: False) Unlike most of the other flags, you do not need to provide a value after this flag. If you do not include the flag, it will be false. If you do include the flag, it will be true.
```
These are the main ones. Run `python run.py --help` to the rest
```
python run.py run_id=AgentCool seed=19 
env_args.record_chamber=false
```

### Testing
Testing is nearly identical to training except you will set 
the testing
flag as true and in most cases include a path to the 
distractor stimulus. Test runs also take much shorter. You should test your agent across multiple test_video stimuli including a case where the test stimuli is left blank (called a rest trial).
```
python run.py test=true env_args.test_video=/path/to/video
```
Below is a list of the additional env_args options that you'll need for testing:
```
--test_video VIDEO_PATH
                      Path to the video containing novel test object. This video will be played on the left monitor first. The video needs to be a webm file if you are using Linux/Unix. (default: None)
```

### Run ml agents on the server (instead of running locally, as above)
1. Basic Unix Commands (for a more exhaustive set of basic Unix commands see https://kb.iu.edu/d/afsk)
     * cd - "change directory," changes your current directory location (note: cd .. moves to parent directory of current working directory)
     * ls - "list," lists the files stored in a directory
     * mkdir - "make directory," makes a new subdirectory in current working directory
     * pwd - "print working directory," reports the current directory path
     * scp - "secure copy" a copy program to transfer files across servers.
2. Git clone this repo onto the server as described in the main README.
3. Transfer executable to server using `scp`.
4. Start a screen by running the command
```screen```
4. Check for available GPUs on server. The one with no process is the one you should use.
```nvidia-smi```
5. Set desired GPU to be accessed by CUDA
```
export VISIBLE_CUDA_DEVICES=Device#
```
6. Follow steps above for running locally but adjust necessary file names.