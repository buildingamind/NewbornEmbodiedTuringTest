#Stable baselines is a well-maintained rl library
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import os #Used for model saving and loading

#Agent class as specified in the config file. Models are stored as files rather than
#being kept in memory for performance reasons.
class Agent:
    def __init__(self, agent_id="Default Agent", reward="supervised", path="./Brains", **kwargs):
        self.reward = reward
        self.id = agent_id
        self.model = None
        
        #If path does not exist, create it as a directory
        if not os.path.exists(path):
            os.makedirs(path)

        #If path is a saved model assign to path
        if os.path.isfile(path):
            self.path = path
        else:
            #If path is a directory create a file in the directory name after the agent
            self.path = os.path.join(path, self.id)

    #Train an agent. Still need to allow exploration wrappers and non PPO rl algos.
    def train(self, env, eps):
        steps = env.steps_from_eps(eps)
        if self.reward == "supervised":
            self.model = PPO("CNNPolicy", env, verbose=1)
        else:
            print("Please use the supervised reward until I implement rlexplore correctly.")
            return
        self.model.learn(total_timesteps=steps)
        self.save()
        del self.model
        self.model = None

    #Test the agent in the given environment for the set number of steps
    def test(self, env, eps):
        self.load()
        if self.model == None:
            print("Usage Error: model is not specified either train a new model or load a trained model")
            return
        
        #Run the testing
        steps = env.steps_from_eps(eps)
        obs = env.reset()
        for i in range(steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
        del self.model
        self.model = None

    #Saves brains to the specified path
    def save(self, path=None):
        if path is None:
            path = self.path
        self.model.save(path)

    #Load brains from the file
    def load(self, path=None):
        if path == None:
            path = self.path
        if not os.path.exists(path):
            print(f"Usage Error: The path {path} does not exist")
            self.model = None
            return
        self.model = PPO.load(path)


def intrinsic_train(model,ep_steps,episodes,envs,module):
    _, callback = model._setup_learn(total_timesteps=ep_steps*episodes, eval_env=None)

    for i in range(episodes):
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=ep_steps,
            callback=callback
        )
        # Compute intrinsic rewards.
        intrinsic_rewards = module.compute_irs(
            buffer=model.rollout_buffer,
            time_steps=i * ep_steps * len(envs))
        model.rollout_buffer.rewards = intrinsic_rewards
        # Update policy using the currently gathered rollout buffer.
        model.train()