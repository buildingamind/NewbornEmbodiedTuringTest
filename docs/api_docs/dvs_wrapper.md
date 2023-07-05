#


## DVSWrapper
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L13)
```python 
DVSWrapper(
   env, change_threshold = 60, kernel_size = (3, 3), sigma = 1
)
```




**Methods:**


### .create_grayscale
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L29)
```python
.create_grayscale(
   image
)
```


### .gaussianDiff
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L33)
```python
.gaussianDiff(
   previous, current
)
```


### .observation
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L45)
```python
.observation(
   obs
)
```


### .threshold
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L55)
```python
.threshold(
   change
)
```


### .reset
[source](/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/env_wrapper/dvs_wrapper.py/#L62)
```python
.reset(
   **kwargs
)
```

