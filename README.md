# Unity environment for ChickAI: virtual controlled-rearing experiments
This is a collection of tools for the simulating of virtual agents using the Building 
a Mind paradigm. The agents
generated and studied through this pipeline can be compared directly to real chicks 
recorded by the Building a Mind
Lab. This pipeline provides all necessary components for simulating and replicating 
embodied models from our lab.

## Sub-Directories

* `Unity`: Contains a Unity project which is a virtual replication of the VR Chambers 
used in contolled-rearing studies from the lab.
* `Executables`: Contains pre-compiled executable versions of the Unity project. The 
folder is also where the scripts will look for the proper environments.
* `Scripts`: Contains the code for running simulations of development, testing agents,
and analyzing generated data.

## Projects
The table below lists the projects and publications using this pipeline.
| Project/Publication | Date | Authors                                | Branch | Webpage |
|---------------------|------|----------------------------------------|--------|---------|
| Imprinting            |      | Donsuk Lee, Samantha Wood, Justin Wood |        |         |
| Binding             |      | Donsuk Lee, Justin Wood                |        |         |



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
