## A newborn embodied Turing test for visual parsing

Manju Garimella, Denizhan Pak, Lalit Pandey, Justin N. Wood, & Samantha M. W. Wood


https://github.com/buildingamind/pipeline_embodied/assets/1686251/839bd04c-8853-44c4-b275-5e61413a3904


### Abstract

*Newborn brains exhibit remarkable abilities in rapid and generative learning, including the ability to parse objects from backgrounds and recognize those objects across substantial changes to their appearance (i.e., novel backgrounds and novel viewing angles). How can we build machines that can learn as efficiently as newborns? To accurately compare biological and artificial intelligence, researchers need to provide machines with the same training data that an organism has experienced since birth. Here, we present an experimental benchmark that enables researchers to raise artificial agents in the same controlled-rearing environments as newborn chicks. First, we raised newborn chicks in controlled environments with visual access to only a single object on a single background and tested their ability to recognize their object across novel viewing conditions. Then, we performed “digital twin” experiments in which we reared a variety of artificial neural networks in virtual environments that mimicked the rearing conditions of the chicks and measured whether they exhibited the same object recognition behavior as the newborn chicks. We found that biological chicks developed background-invariant object recognition, while the artificial chicks developed background-dependent recognition. Our benchmark exposes the limitations of current unsupervised and supervised algorithms in achieving the learning abilities of newborn animals. Ultimately, we anticipate that this approach will contribute to the development of AI systems that can learn with the same efficiency as newborn animals.*

### Experiment Design

- VR chambers were equipped with two display walls (LCD monitors) for displaying object stimuli.
- During the Training Phase, artificial chicks were reared in an environment containing a single 3D object rotating a full 360° around a horizontal axis in front of a naturalistic background scene. The object made a full rotation every 15s.
- During the Test Phase, the VR chambers measured the artificial chicks’ imprinting response and object recognition performance. The “imprinting trials” measured whether the chicks developed an imprinting response.  The “test trials” measured the aritifical chicks’ ability to visually parse and recognize their imprinted object. During these trials, the imprinted object was presented on one display wall and an unfamiliar object was presented on the other display wall. Across the test trials, the objects were presented on all possible combinations of the three background scenes (Background 1 vs.Background 1, Background 1 vs. Background 2, Background 1 vs.Background 3, etc.).

### Arguments

#### Train configuration

```
agent_count: 1
run_id:ship_backgroundA_exp
log_path: data/ship_backgroundA_exp
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
  env_path: data/executables/parsing_benchmark/parsing.x86_64
  log_path: data/ship_backgroundA_exp/Env_Logs
  rec_path: data/ship_backgroundA_exp/Recordings/
  record_chamber: false
  record_agent: false
  recording_frames: 0
```

### Links
