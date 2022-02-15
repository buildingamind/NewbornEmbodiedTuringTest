# Build the ChickAI environment from the Unity editor.
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

