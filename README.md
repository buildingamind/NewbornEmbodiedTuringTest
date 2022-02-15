# Unity environment for ChickAI: virtual controlled-rearing experiments

## Running experiments using `mlagents-learn`
### Setup
#### Part 1: Build the ChickAI environment from the Unity editor.
In this section, you will pull this repository from Github, open the Unity environment, and build the ChickAI environment as an executable.
1. Install Git and/or Github Desktop. If you install Git, you'll be able to interact with Github through the command line. You can download Git using the directions here: https://git-scm.com/downloads. If you install Git Desktop, you can use a GUI to interact with Github. You can install Github Desktop by following the directions here: https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/. For the following steps, I will provide the command line arguments (but you can use the GUI to find the same options in Github Desktop).
2. To download the repository, click the Code button on the chickAI-unity repo. Copy the provided web URL. Then follow the code below to change to directory where you want the repo (denoted here as MY_FOLDER) and then clone the repo.
```
cd MY_FOLDER
git clone URL_YOU_COPIED_GOES_HERE
```
3. Checkout the branch you want to be extra sure that you're using the right branch.
```
cd chickAI-unity
git checkout DESIRED_BRANCH
```
4. Download Unity Hub from https://unity3d.com/get-unity/download
5. Open Unity Hub, click Add button, and choose the directory chickAI-unity. You may also need to download the corresponding Unity Version using Unity Hub as well.
6. Open the chickAI-unity project using Unity Hub.
7. In the Unity Editor, go to Scenes and open the controlled rearing scene. Check the scene to make sure that everything looks correct. Is there a chick agent in a chamber? Is there a camera overhead? Is there a camera attached to the chick agent? Is there an Agent recorder and a Chamber recorder? Are all of the appropriate elements in the inspector enabled? Is the agent using the correct brain setting?
8. Build the executable for your local machine. Go to file --> Build settings. <b>Make sure the controlled rearing scene (and ONLY the controlled rearing scene) are checked.</b> Select your platform. Use the platform for your computer. Always build a local version to test, even if you will ultimately run this on the server. Building a local version of the executable is necessary to make sure that the environment looks right and that the movies are loading on the display walls correctly. Always build a local version first.
9. You may also want to build a version to run on the server. Follow the same instructions as above for building an executable, but choose Linux/Unix as your platform. Do NOT choose the "server build" option in the dialog box.

#### Part 2: Set up your virtual environment
Why use a virtual environment? Virtual environments help keep dependencies required by different projects separate.
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

#### Part 3: Run ml agents on the server (instead of running locally, as above)
1. Basic Unix Commands (for a more exhaustive set of basic Unix commands see https://kb.iu.edu/d/afsk)
     * cd - "change directory," changes your current directory location (note: cd .. moves to parent directory of current working directory)
     * ls - "list," lists the files stored in a directory
     * mkdir - "make directory," makes a new subdirectory in current working directory
     * pwd - "print working directory," reports the current directory path
     * sudo - allows you to run programs with the security privileges of the "superuser"
     * rm - "remove," removes (destroys) a file. (You can enter this command with the -i option, so that you'll be asked to confirm each file deletion.)
     * rmdir - "remove directory," removes an empty directory. If you want to remove a directory with files in it, you'll need to use rm -r within that directory to delete the files first.
     * -r - "recursive," an optional argument for commands like rm, cp, and scp that specifies that the command should be applied to all files in the directory
     * cp - "copy," copies a file, preserving the original and creating an identical copy
     * scp - "secure copy," same as cp, but used to copy files over a network. If you are transferring between local machine and server, use scp
     * chmod - "change mode," changes the permission information associated with a file
     * mv - "move," moves a file. You can use mv not only to change the directory location of a file, but also to rename files. Unlike the cp command, mv will not preserve the original file.
     * ssh - "secure shell," provides a secure encrypted connection between two hosts

2. First, you'll need to move your files over to the server. What do you need to move over? You'll need all the webm files for the experiment. You'll also need all of the relevant executable files: your .x86_64 file, LinuxPlayer_s.debug, UnityPlayer_s.debug, and UnityPlayer.so.
```
# single file secure copy to server
scp LOCAL_FILE_ADDRESS USER@SERVER.luddy.indiana.edu:/home/user/DIRECTORY
# directory secure copy to server
scp -r LOCAL_FILE_ADDRESS USER@SERVER.luddy.indiana.edu:/home/user/DIRECTORY
```

3. Connect to the server
```
ssh YOUR_USERNAME@SERVER.luddy.indiana.edu
```

4. If you haven't already, set up the virtual environment on the server. The server should already have virtualenv installed. Follow the sub-steps below to create a virtual environment for your project.
<p><ul><ul>a) Create a virtual environment. For code below, the directory where you want your virtual environment is denoted as MY_FOLDER, and the name of your virtual environment is denoted as VENV. (You shouldn't use all caps. I'm just using them to draw attention to the part of the code you'll need to personalize.)</br>
<code>
cd MY_FOLDER
python3 -m venv VENV
</code>
</br>
b) Activate your virtual environment</br>
<code>
source VENV/bin/activate
</code>
</br>
c) Install `ml-agents` Python package. (You'll do this while your virtual environment is activated so that your virtual environment will have the correct version of ml-agents.)</br>
<code>
python3 -m pip install mlagents==0.26.0
</code>
</br>
d) Your virtual environment should be activated whenever you want to train and test models with this version of the environment. However, when you are finished with ml-agents, you can deactivate the virtual environment:</br>
<code>
deactivate
</code>
</br>
</ul></ul></p>

5. Check the GPU usage. 
```
nvidia-smi
```
This will show you if anyone else is running on the server, and if so, what GPUs they are using. It will also show you if the X Server is running. The X server acts as a virtual display (so that the game isn't played "headless.") If the X Server is running correctly, the bottom "processes" table will show "/usr/lib/xorg/Xorg" for every GPU. If you don't see the Xorg process, then the X Server needs to be restarted. The command to restart the server is below, but note that you'll need sudo privileges to restart the X Server.
```
sudo /usr/bin/X :0 
```

6. Start a screen. A screen is a virtual shell session launched within your current shell session. You can detach from a screen, and it will continue running on the server, even if you quit Terminal or shut off your device. It's always a good habit to start a screen first because training an agent takes a long time.
```
screen -S SESSION_NAME
```
<p><ul><ul>
  <b>Basic Screen Commands</b>
  <ul><li>Starting a screen: screen -S SESSION_NAME</li>
  <li>Detach from screen: Ctrl-a + d</li>
  <li>Get list of all screens: screen -ls</li>
  <li>Resuming a screen: screen -r SESSION_NAME</li>
  <li>Killing a screen: screen -X -S SESSION_NAME</li>
    <li>Killing the screen you are currently using: Ctrl-a + k</li></ul>
  </br>
  <p>For more screen commands see: https://linuxize.com/post/how-to-use-linux-screen/</p>
</p></ul></ul>  


7. Activate your virtual environment. Make sure to do this in your screen session.
```
source VENV/bin/activate
```

8. Make Ubuntu use X Server for display.
```
export DISPLAY=:0
```

9. Run the training for ml-agents. When you do this, you'll usually need to add an additional argument before --env-args to specify your baseport. If you try to run on the same baseport as another process that is currently running, it won't work. 
Another reminder here (as above) that the address for your webm files must be ABSOLUTE, rather than relative for the environment to find the file.
```
# Generic command
mlagents-learn CONFIG_PATH --env ENV_PATH --run-id RUN_ID --base-port BASE_PORT --env-args ENV_ARGS

# Example:
mlagents-learn config.yaml --env env_builds/Sep_build/chickAI_linux.x86_64 --run-id agent3 --base-port 6000 --env-args --input-resolution 96 --imprint-video /home/sw113/chickAI/movies/SPM/imprint1.webm --record-chamber --record-agent
```

### Videos
You can play a video on each of the two monitors inside the chamber. The `imprint-video` will be played on the right monitor and the `test-video` will be played on the left monitor in the first episode. Then, at the beginning of each new episode, videos will be swapped between the two monitors.

If you are running Unity on a Linux system, you must convert videos to `webm` format.
