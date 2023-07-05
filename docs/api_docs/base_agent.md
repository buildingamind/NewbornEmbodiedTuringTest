#


## BaseAgent
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L24)
```python 
BaseAgent(
   agent_id = 'DefaultAgent', log_path = './Brains', **kwargs
)
```




**Methods:**


### .train
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L64)
```python
.train(
   env, eps
)
```


### .test
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L68)
```python
.test(
   env, eps, record_prefix = 'rest'
)
```

---
Test the agent in the given environment for the set number of steps


**Args**

* **env**  : gym environment wrapper
* **eps**  : number of test episodes
* **record_prefix** (str, optional) : recording file name prefix


### .save
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L108)
```python
.save(
   path: Optional[str] = None
)
```

---
Save agent prains to the specified path


**Args**

* **path** (str) : Path value to save the model


### .load
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L124)
```python
.load(
   path = None
)
```

---
Load the model from the specified path


**Args**

* **path** (str) : model saved path. Defaults to None.


### .check_env
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L137)
```python
.check_env(
   env
)
```

---
Check environment


**Args**

* **env** (vector environment) : vector env check for correctness


**Raises**

* **Exception**  : raise exception if env check fails


**Returns**

* **bool**  : env check is successful or failed


### .plot_results
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/common/base_agent.py/#L157)
```python
.plot_results(
   steps: int, plot_name = 'chickai-train'
)
```

---
Generate reward plot for training


**Args**

* **steps** (int) : number of training steps
* **plot_name** (str, optional) : Name of the reward plot. Defaults to "chickai-train".

