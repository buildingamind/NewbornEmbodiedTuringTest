# A Newborn Embodied Turing Test for View-Invariant Recognition

Denizhan Pak, Donsuk Lee, Samantha M. W. Wood & Justin N. Wood



https://github.com/buildingamind/pipeline_embodied/assets/1686251/2fed4649-b4d6-4c93-813c-cd040a92c8cb


## Abstract

*Recent progress in artificial intelligence has renewed interest in building machines that learn like animals. Almost all of the work comparing learning across biological and artificial systems comes from studies where animals and machines received different training data, obscuring whether differences between animals and machines emerged from differences in learning mechanisms versus training data. We present an experimental approach—a “newborn embodied Turing Test”—that allows newborn animals and machines to be raised in the same environments and tested with the same tasks, permitting direct comparison of their learning abilities. To make this platform, we first collected controlled-rearing data from newborn chicks, then performed “digital twin” experiments in which machines were raised in virtual environments that mimicked the rearing conditions of the chicks. We found that (1) machines (deep reinforcement learning agents with intrinsic motivation) can spontaneously develop visually guided preference behavior, akin to imprinting in newborn chicks, and (2) machines are still far from newborn-level performance on object recognition tasks. Almost all of the chicks developed view-invariant object recognition, whereas the machines tended to develop view-dependent recognition. The learning outcomes were also far more constrained in the chicks versus machines. Ultimately, we anticipate that this approach will help researchers develop embodied AI systems that learn like newborn animals.*

## Experiment Design

- VR chambers were equipped with two display walls (LCD monitors) for displaying object stimuli.
- During the Training Phase, artificial chicks were reared in an environment containing a single 3D object rotating 15° around a vertical axis in front of a blank background scene. The object made a full rotation every 3s. Agents can be imprinted to one of 4 possible conditions: side and front views of the Fork object or side and front views of the ship object.
- During the Test Phase, the VR chambers measured the artificial chicks’ imprinting response and object recognition performance. The “imprinting trials” measured whether the chicks developed an imprinting response.  The “test trials” measured the aritifical chicks’ ability to visually discriminate their imprinted object. During these trials, the imprinted object, rotated at an alternate angle to the imprint condition, was presented on one display wall and an unfamiliar object was presented on the other display wall, the angle of which was either the same as the imprint condition (fixed trials) or matched to the viewpoint in the test condition (matched trials).

## Arguments

### Train configuration

```
agent_count: 1
run_id:ship_front_exp
log_path: data/ship_front_exp
mode: full
train_eps: 1000
test_eps: 40
cuda: 0
Agent:
  reward: supervised
  encoder: small
Environment:
  use_ship: true
  side_view: false
  background: A
  base_port: 5100
  env_path: data/executables/viewpoint_benchmark/viewpoint.x86_64
  log_path: data/ship_front_exp/Env_Logs
  rec_path: data/ship_front_exp/Recordings/
  record_chamber: false
  record_agent: false
  recording_frames: 0
```

## Executables

[Exectuable can be found here](https://origins.luddy.indiana.edu/unity/executables/).
