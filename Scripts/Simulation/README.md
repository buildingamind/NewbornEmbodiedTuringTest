# chickAI-experiments
## Running experiments using `mlagents-learn`
### Setup
#### Part 1: Set up your virtual environment
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
2. Create a virtual environment. For code below, the directory where you want your virtual environment is denoted as MY_FOLDER, and the name of your virtual environment is denoted as VENV. (You shouldn't use all caps. I'm just using them to draw attention to the part of the code you'll need to personalize.)

MAC / LINUX:
```
cd MY_FOLDER
python3 -m venv VENV
```
WINDOWS:
```
cd MY_FOLDER
py -m venv VENV
```
3. Activate your virtual environment

MAC / LINUX:
```
source VENV/bin/activate
```
WINDOWS:
```
.\VENV\Scripts\activate
```
4. Install `ml-agents` Python package. (You'll do this while your virtual environment is activated so that your virtual environment will have the correct version of ml-agents.)

MAC / LINUX:
```
python3 -m pip install mlagents==0.26.0
```
WINDOWS:
```
pip3 install torch==1.8.0
python3 -m pip install mlagents==0.26.0
```
5. Your virtual environment should be activated whenever you want to train and test models with this version of the environment. However, when you are finished with ml-agents, you can deactivate the virtual environment:
```
deactivate
```

#### Part 2: Compile Unity Environment
Follow instructions in the Unity subdirectory to compile the unity executables for the environments.

#### Part 3: Run Experiments
Follow instructions in scripts directory to run the individual experiments.
### Training
1. Prepare a [trainer configuration file](https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Training-Configuration-File.md) and adjust hyperparameters in the config file.

2. Run `mlagents-learn` command with the config file. In the sample command below, the path to your configuration file is denoted as CONFIG_PATH, the path to your executable environment is denoted as ENV_PATH. You will also need to generate an ID for this brain, which is denoted as RUN_ID. Finally, you will need to supply a bunch of environment arguments, which are denoted as ENV_ARGS and described in detail below.
```
mlagents-learn CONFIG_PATH --env ENV_PATH --run-id RUN_ID --env-args ENV_ARGS
```
* Replace `CONFIG_PATH`, `ENV_PATH`, `RUN_ID` with appropriate values.
* You can configure the environment by passing options after `--env-args`. For example, to set the length of episodes to 2000, replace `ENV_ARGS` with `--episode-steps=2000`. Below is the list of `--env-arg` options that are relevant for training:
```
--input-resolution INPUT_RESOLUTION
                      Resolution of visual inputs to agent. (default: 64)
--episode-steps EPISODE_STEPS
                      Number of steps per episode. (default: 1000)
--imprint-video VIDEO_PATH
                      ABSOLUTE path to the video containing imprinting object. If you specify a relative path (e.g., /my_video_dir), nothing will play on the display walls. You need to use an absolute path (e.g., /home/user/research/chickAI/my_video_dir). This video will be played on the right monitor first. The video needs to be a webm file if you are using Linux/Unix. (default: None)
--log-dir LOG_DIR     Directory to save environment logs. You NEED to provide a log-dir. Seriously. You won't get a log of where the agent is without it. (default: None)
--record-agent
                      Record the agent's camera if true. (default: False) Unlike most of the other flags, you do not need to provide a value after this flag. If you do not include the flag, it will be false. If you do include the flag, it will be true.        
--record-chamber
                      Record the chamber camera if true. (default: False) Unlike most of the other flags, you do not need to provide a value after this flag. If you do not include the flag, it will be false. If you do include the flag, it will be true.
```
Here is an example of what the command will look like with actual flags, paths, etc.:
```
mlagents-learn config.yaml --env chickAI.exe --run-id parsing_agent1 --env-args --input-resolution 96 --imprint-video ~/folder/folder/parsing_videos/imprintA.webm --log-dir parsing_agent1_imprinting
```

### Testing
When testing, you will need to provide some additional flags. Specifically, you will need to provide a resume and an inference flag before the ENV_ARGS, and you will need to provide the test-mode flag as one of your ENV_ARGS. You'll also need to provide the video path to the test video (as well as the imprinting video) by using the test-video ENV_ARG. The RUN_ID should be identical to the agent you previously trained (that's how ml-agents knows which brain to load), but your LOG_DIR should be unique (so that you don't save over the training log-dir or any other test log-dirs). It should look something like this:
```
mlagents-learn CONFIG_PATH --env ENV_PATH --run-id RUN_ID --resume --inference --env-args ENV_ARGS --test-video MY_TEST_VIDEO.WEBM --test-mode
```
Below is a list of the additional --env-arg options that you'll need for testing:
```
--test-video VIDEO_PATH
                      Path to the video containing novel test object. This video will be played on the left monitor first. The video needs to be a webm file if you are using Linux/Unix. (default: None)
--test-mode 
                      Run environment in test mode if True. (default: False) Unlike most of the other flags, you do not need to provide a value after this flag. If you do not include the flag, it will be false. If you do include the flag, it will be true.
```
Here is an example of what the command will look like with actual flags, paths, etc.:
```
mlagents-learn config.yaml --env chickAI.exe --run-id parsing_agent1 --resume --inference --env-args --input-resolution 96 --imprint-video ~/folder/folder/parsing_videos/imprintA.webm --test-video ~/folder/folder/parsing_videos/testB.webm  --log-dir parsing_agent1_testCondition1 --record-agent --record-chamber --test-mode
```

### Run ml agents on the server (instead of running locally, as above)
1. Basic Unix Commands (for a more exhaustive set of basic Unix commands see https://kb.iu.edu/d/afsk)
     * cd - "change directory," changes your current directory location (note: cd .. moves to parent directory of current working directory)
     * ls - "list," lists the files stored in a directory
     * mkdir - "make directory," makes a new subdirectory in current working directory
     * pwd - "print working directory," reports the current directory path
