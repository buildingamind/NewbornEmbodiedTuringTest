# Random Seeding in NETTs

## Training

When training, the agent number (brain number) is used as the seed for all random number generators. For example, if `brain.num_brains==5`, there would be 5 agents each with the seeds 1, 2, 3, 4, and 5 respectively. This seed is used in generating random numbers that are consistent between runs. As long as the seed is the same, all random aspects of the training will be the same, assuming that the length of training is the same. There are two main sources of randomness during training:

1. Model seed - The subset of steps assigned to each mini-batch from the replay buffer.

    - This is defined during intialization of the model in the Brain component.
    - For example, if the replay buffer has 4 steps and the mini-batch size is 2, the mini-batches could be [1,2] and [3,4], [1,3] and [2,4], [1,4] and [2,3], or any of these with their orders altered. The seed determines which of these combinations is used.

2. Environment seed - The initial placement of agents in the environment at the start of each episode.

    - This is defined during initialization of the Unity Environment in the Environment component.
    - During Training, the position and orientation of the agent is randomized at the start of each episode.  

## Testing

During Testing, nothing is random, so seeding does not impact the results.
