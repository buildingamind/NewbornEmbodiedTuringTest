# Unity environment for ChickAI: virtual controlled-rearing experiments

## Running experiments using `mlagents-learn`
### Setup
#### Part 1: Set up your virtual environment
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

