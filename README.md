# **Unity environment for ChickAI: virtual controlled-rearing experiments**

This is a collection of tools for simulating virtual agents under controlled-rearing conditions. The agents
generated and studied through this pipeline can be compared directly to real chicks recorded by the **[**Building a Mind
Lab**](http://buildingamind.com/)**. This pipeline provides all necessary components for simulating and replicating embodied models from the lab.

The figure below shows the experiment setup for the three experiments discussed in the guide.

<img src="docs/digital_twin.jpg" alt="Digital Twin" style="zoom:35%;" />

## **How to Use this Repository**

This directory provides three components for building embodied virtual agents. These are a video game which serves as a virtual world, a set of programs to run experiments in the virtual world, and a set of programs to visualize the data coming from the experiments.

## **Directory Structure**

**Following the directory structure of the code.**

```

├── docs
├── mkdocs.yml
├── README.md
├── notebooks
│   ├── Getting Started.ipynb
├── src/netts
│   ├── body
│   └── brain
│   ├── environment
│   └── utils
│   ├── nett.py
│   └── __init__.py
└── tests

```

* `src/netts`**: **Contains the code for running experiments with simulated agents. Following is the structure of `src/netts` folder:
* `tests`: Contains unit tests
* `docs`: Contains project documentation 

## **Getting Started**

In this section, you will learn to use the repository to benchmark your first embodied agent with NETTs!

**NOTE:** The dependency `mlagents==1.0.0` is not compatible with Apple Silicon (M1, M2, etc.) devices. Please use a different device to run the codebase.

### **Codebase Installation**

1. **(Highly Recommended)** [create and configure a virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda "Link for how to set-up a virtual env")
   **steps described below :**
   ```
   conda create -n netts python=3.10.12
   conda activate netts
   ```
2. Install the needed versions of `setuptools` and `pip`:
   ```
   pip install setuptools==65.5.0 pip==21
   ```
   **NOTE:** More recent versions of `setuptools` and `pip` are incompatible with the multiple dependencies due to a subdependency `gym==0.21`. More information about this issue can be found [here](https://github.com/openai/gym/issues/3176#issuecomment-1560026649)
3. Install the repository using `pip`:
   ```
   pip install nett-benchmarks
   ```

**NOTE:** If not installing in a virtual environment, the install might fail because of conflicting dependency versions (see note under Step #2). `gym==0.21` has a dependency on older versions of `numpy`, specifically `1.21.2` or below. Ensure that these requirements are met in your environment before proceeding. 

### **Running a NETT**

After having followed steps above, NETTs can be run with a few lines of code:

1. Download the executable from the hosted webpage or build your own executable following the steps mentioned in {placeholder}. Additionally, a video walkthrough of creating an executable from scratch is made available {here}. 

2. If you are running the code on a machine without a display (i.e. a server), you will need to install a virtual display. This can be done using the following command:
```
export DISPLAY=:0
```

3. The package defines three components: `Brain`, `Body` and `Environment`. For a detailed description of the division of responsibilities of each component, please refer to the {documentation}. 
```
from netts import Brain, Body, Environment
from netts import NETT
```

4. Define each component as required. Let’s start with the `Brain`. This component holds all the parts concerned with “learning”. This includes the architecture itself along with its many parameters, as well as the reward function and learning algorithm, such as `PPO`. The package is designed to make each component flexible. Specifically, each constituent of the Brain such as the encoder, policy networks or the reward function can be customized. The training and testing loops can be customized by inheriting the class and overriding them. This may be necessary in specialized cases, such as running on customized hardware such as TPU and IPUs. More details can be found in the documentation. 

Consider a rather simple definition: 
```
brain = Brain(policy="CnnPolicy", algorithm="PPO")
```

This defines the policy network and the learning algorithm, the rest of the arguments are left to defaults. Please refer to the repository for a complete list of supported arguments and customizations. This document will be updated with appropriate links to tutorials planned as part of future releases too. Note that the repository uses modules from `stable-baselines3` underneath and the value specified as arguments directly correspond with it. 

5. Next, define the `Body`. Technically, the body is a medium through which all information passes through before reaching the brain. Hence, this component is primarily concerned with the application of `gym.Wrappers` (such as the DVS wrapper) that modify the information from the environment before they “reach” the brain for processing. 

Consider a simple definition: 
```
body = Body(type="basic", dvs=False, wrappers=None)
```

In this example, we do not pass any wrappers. Or, we let the information from the environment reach the brain "as is". 

While the `Body` abstraction is thin and mostly conceptual as of now, different types of agents (`two-eyed`, `rag-doll`) with complex action spaces are planned which lead to `Body` taking on more sophisticated customization. 

6. The `Environment` component constructs a live environment (via the Python `mlagents` library) which is then wrapped inside a `Gym` environment. Since this is part of the environment initialization, the wrapping is not included as part of the `Body` component, but kept within the `Environment`. The definition takes an executable path which must be available on the system. 
```
# a valid Unity executable downloaded from {} must be present at `executable_path`
environment = Environment(config="identityandview", executable_path=executable_path)
```

7. For a full list of the NETT configs available, one can simply do the following. 
```
from netts.environment.configs import list_configs
list_configs()
```

Similar analogues for listing the encoders, policies, algorithms are also available: 
```
from netts.brain import list_algorithms, list_policies, list_encoders
```

8. In order to orchestrate the benchmarking, all three components are brought together under one umbrella, the `NETT` object. This allows for storing details of the runs, serving as a reproducible artifact, automatic distribution of runs with different imprinting conditions, central logging, among other things. 
```
nett = NETT(brain=brain, body=body, environment=environment)
```

9. The created `nett` instance has the `.run()` method which carries out the execution. All the obvious runtime parameters with respect to benchmarking such as number of brains, train and test episodes, device(s) to be used for execution, steps per episode etc, are accepted as input to this function:

```
run = netts.run(dir=run_dir, num_brains=5, trains_eps=10)
```

10. Internally, the function automatically creates a “task” list and assigns parallel processes for workers that execute the “jobs”. The result is the `run` object that can be used to inquire about the status of each of the “jobs” along with information about the device, mode etc. This is provided using the `.status()` method: 
```
nett.status(job_sheet)
```

Note that the `.run()` call executes in the background and does not block until its completion. This means the `run`` object is returned immediately and can be checked on a regular basis (programmatically or manually) to infer the completion of the run (or individual jobs).


### **Running Standard Analysis**

TO BE UPDATED.

## **Experiment Configuration**

More information related to details on the experiment can be found on following pages.

* [**Parsing Experiment**](docs/Parsing.md)
* [**ViewPoint Experiment**](docs/ViewInvariant.md)