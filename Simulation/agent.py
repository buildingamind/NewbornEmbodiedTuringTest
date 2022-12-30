#Stable baselines is a well-maintained rl library
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os #Used for model saving and loading

#Agent class as specified in the config file. Models are stored as files rather than
#being kept in memory for performance reasons.
class Agent:
    def __init__(self, agent_id="Default Agent", reward="supervised", log_path="./Brains", **kwargs):
        self.reward = reward
        self.id = agent_id
        self.model = None
        
        #If path does not exist, create it as a directory
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        #If path is a saved model assign to path
        if os.path.isfile(log_path):
            self.path = log_path
        else:
            #If path is a directory create a file in the directory name after the agent
            self.path = os.path.join(log_path, self.id)

    #Train an agent. Still need to allow exploration wrappers and non PPO rl algos.
    def train(self, env, eps):
        steps = env.steps_from_eps(eps)
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        if self.reward == "supervised":
            self.model = PPO("CnnPolicy", envs, verbose=1)
        else:
            print("Please use the supervised reward until I implement rlexplore correctly.")
            return
        self.model.learn(total_timesteps=steps)
        self.save()
        del self.model
        self.model = None
    
    def train_intrinsic(self, env, eps):
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        re3 = RE3(obs_shape=envs.observation_space.shape, 
                action_shape=envs.action_space.shape, 
                device=device, latent_dim=128, beta=1e-2, kappa=1e-5)
        #Need to figure out how to make this generic and use it.

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


