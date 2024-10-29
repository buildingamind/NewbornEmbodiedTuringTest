"""
Gym wrapper for a Unity ML-Agents environment. 
Currently does not support:
-multiple brains in the env
-flattened branch
"""

import numpy as np

import gym
from gym import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import DecisionSteps #BatchedStepResult

class MultiAgentEnvException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass

class MultiAgentEnv(gym.Env):
    def __init__(self, environment_filename, additional_args=[], base_port=5005, seed=0):
        """
        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment. Default 0.
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done). Default False.
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode. Default False.
        """
        self._env = UnityEnvironment(
            environment_filename,
            side_channels=[EngineConfigurationChannel()],
            additional_args=additional_args,
            base_port=base_port, seed=seed)
        
        # if not self._env.get_agent_groups():
        agent_groups = list(self._env._env_specs.keys())
        if not agent_groups:
            self._env.step()
        
        self._multiagent = True
        self.brain_name = agent_groups[0] #self._env.get_agent_groups()[0]
        self.group_spec = self._env._env_specs[self.brain_name] #self._env.get_agent_group_spec(self.brain_name)
        
        self._env.reset()
        step_result = self._env._env_state[self.brain_name] #self._env.get_step_result(self.brain_name)
        
        self._previous_step_result = step_result
        self._previous_new_id_order = list(range(step_result.n_agents()))
        self._previous_done_agents = 0
        
        self._agents_id = list(self._previous_step_result.agent_id)
        
        self.n_agents = len(self._agents_id)
        self._n_actions_to_send = self.n_agents
        
        # Check brain configuration
        if len(list(self._env._env_specs.keys())) != 1:
            raise MultiAgentEnvException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym."
            )
        
        if self.group_spec.is_action_discrete():
            branches = self.group_spec.discrete_action_branches
            if self.group_spec.action_shape == 1:
                print("Action is Descrete")
                self.action_space = spaces.Discrete(branches[0])
            else:
                print("Action is MultiDiscrete")
                self.action_space = spaces.MultiDiscrete(branches)
        else:
            print("Action is Box")
            high = np.array([1] * self.group_spec.action_shape)
            self.action_space = spaces.Box(-high, high, dtype=np.float32)
            
        high = np.array([np.inf] * self._get_vec_obs_size())
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        
        step_result = self._step(True)
        
        if self._multiagent:
            return self._multi_step(step_result)[0]
        else:
            return self._single_step(step_result)[0]
        
    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()
        
    def step(self, action):
        """Run one timestep of the environment's dynamics. 
        Accepts an action and returns a tuple (observation, reward, done, info).
        In the case of multi-agent environments, these are lists.
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            maxstep (boolean/list): whether the episode has ended because the agent ran out of time.
        """
        
        action = self._sanitize_action(action)
        self._env.set_actions(self.brain_name, action)
        step_result = self._step()
        
        if self._multiagent:
            return self._multi_step(step_result)
        else:
            return self._single_step(step_result)
    
    def _sanitize_step_result(self, step_result):
        """
        Takes as input a BatchedStepResult returned from mlagents_envs and cleans it in order to send back informations about agents in always the same order.
        This order is given by self._agents_id. 2 possible cases :
        1) No agents terminated on the new timestep
        2) One or more agents aterminated on the new timestep
        
        If 1), the step_result doesnt need to be modified.
        If 2), modifications need to be made on the step_result.
        
        For some reasons, when an agent is done, mlagents_envs returns in step_result informations about the done agent as well as informations about a new agent,
        added because the agent terminated. We want to treat these two agents as the same agent. Furthermore, the information about the new agent is located at a
        specific position in the step_result.
        To illustrate this, let's say we receive this step_result at timestep t: [0, 1, 2] and agent 1 terminated at t+1. We will receive : [1, 0, 3, 2].
        Few things happen here:
        -the done agent (1) is put at the first place of the step_result at t+1.
        -the new agent (3) is put at the index that agent 1 was on the last timestep, + 1.
        
        In fact, we can generalize this in the case of n agents being done at timestep t+1: the index of a new agent corresponding to a certain agent which just
        terminated is the index of the agent that terminated on the last timestep + n - m, n being the number of done agents at timestep t+1, and m the number of done agents at timestep t.
        Why n ? Because n agents were "pushed" at the beginning of step_result thus we need to include them to access the new agent.
        Why m ? If agents were done at timestep t, they have been removed from the step_result of timestep t+1. We thus need to substract them to access the new agent
        (it is easier to see this if you take a pencil and a paper and simulate the process)
        
        So, in order to return a step_result which is "sanitized" i.e. return a step_result with the same order as self._agents_id, we need to do a few things :
        -create new_id_order: list of index corresponding to locations of self._agents_id 
                              (if new_id_order = [2, 0, 1], then id of index 0 in self._agents_id is located at index 2 in step_result, id of index 1 at index 0, and id of index 2 at index 1)
        -create index_gym_id_done: list of index of agent ids in self._agents_id that terminated at current timestep (done=True)
        -replace agents which are done by their successor agents in self._agents_id and create agents_new_id, a list of the new agents.
         To do that, we use the previous step result to locate the position of each done agent. We then deduce the position of their successor (index + n, as said above).
         Once we have the position of their successoir, we access their id.
        -create new step_result, which is composed of:
            -obs: observations of all agents. NOTE: mlagents_envs doesnt provide the last observation (S_T) of a done agent, so we return instead the first observation
                  of its successor.
            -rewards: rewards obtained by all agents. We return step_result.reward[new_id_order] in order to rank them in the right order.
            -dones: whether or not agent termianted on the timestep. We return done=step_result.done[new_id_order] in order to rank them in the right order.
            -max_step: whether or not the agent terminated by running out of timesteps. We return step_result.max_step[new_id_order] in order to rank them in the right order.
            -agent_id: list of agent ids.
            -action_mask: not implemented, so None.
            
        """
        
        #Case 1): simply return step_result
        # in this case: no done agents, the order of step_result is thus the same as the order of self._agents_id
        # so we can set new_id_order to be range(n) ([0, 1, 2, ..., n])
        if len(self._agents_id) == step_result.n_agents():
            self._previous_step_result = step_result
            self._previous_new_id_order = list(range(len(self._agents_id)))
            self._previous_done_agents = 0
            
            return step_result
        
        #Case 2): modify step_result
        
        new_id_order = []
        for agent_id in self._agents_id:
            agent_id_index_step_result = list(step_result.agent_id).index(agent_id)
            new_id_order.append(agent_id_index_step_result)
        
        index_gym_id_done = []
        for index, agent_id in enumerate(step_result.agent_id):
            if step_result.done[index]:
                index_gym_id_done.append(self._agents_id.index(agent_id))
            
        agents_new_id = []
        #2 things here : -replace in self._agents_id the ids of dones agents by ids of their successor.
        #                -create agents_new_id, a list of the successors' ids.
        for index_id_done in index_gym_id_done:
            index_new_agent = self._previous_new_id_order[index_id_done] + len(index_gym_id_done) - self._previous_done_agents
            self._agents_id[index_id_done] = list(step_result.agent_id)[index_new_agent]
            agents_new_id.append(list(step_result.agent_id)[index_new_agent])
        
        new_obs = []
        for index, agent_id in enumerate(self._agents_id):
            if agent_id in agents_new_id:
                new_obs.append(step_result.obs[0][self._previous_new_id_order[index_gym_id_done[agents_new_id.index(agent_id)]] + len(index_gym_id_done) - self._previous_done_agents])
            else:
                new_obs.append(step_result.obs[0][new_id_order[index]])
        new_obs = [np.array(new_obs)]
            
        self._previous_step_result = step_result
        self._previous_new_id_order = new_id_order
        self._previous_done_agents = len(index_gym_id_done)
        
        # if step_result.done[new_id_order]:
        new_step_result = DecisionSteps(obs=new_obs, 
                                            reward=step_result.reward[new_id_order], 
                                            # done=step_result.done[new_id_order], 
                                            # max_step=step_result.max_step[new_id_order], 
                                            group_id = np.zeros(len(self._agents_id)),
                                            group_reward = np.zeros(len(self._agents_id)),
                                            agent_id=step_result.agent_id[new_id_order], 
                                            action_mask=None)
        
        return new_step_result
    
    def _sanitize_action(self, action):
        if self._n_actions_to_send > len(self._agents_id):
            [action.insert(0, 0) for i in range(self._n_actions_to_send - len(action))]
            return np.array(action).reshape((self._n_actions_to_send, self.group_spec.action_size))
        else:
            return np.array(action).reshape((len(self._agents_id), self.group_spec.action_size))
    
    def _step(self, reset=False):
        if reset:
            self._env.reset()
        else:
            self._env.step()
            
        step_result = self._env.get_step_result(self.brain_name)
        
        self._n_actions_to_send = step_result.n_agents()
        
        while step_result.n_agents() - sum(step_result.done) < self.n_agents:
            self._env.step()
            step_result_bis = self._env.get_step_result(self.brain_name)
        
            step_result.obs[0] = np.append(step_result.obs[0], step_result_bis.obs[0], axis=0)
            step_result.reward = np.append(step_result.reward, step_result_bis.reward)
            step_result.done = np.append(step_result.done, step_result_bis.done)
            step_result.max_step = np.append(step_result.max_step, step_result_bis.max_step)
            step_result.agent_id = np.append(step_result.agent_id, step_result_bis.agent_id)
            step_result.action_mask = np.append(step_result.action_mask, step_result_bis.action_mask)
            
            self._n_actions_to_send = step_result_bis.n_agents()
        
        return self._sanitize_step_result(step_result)
    
    def _single_step(self, step_result):
        obs = step_result.obs[0]
        if len(obs.shape) == 2:
            obs = np.concatenate(obs)
        return (obs, step_result.reward[0], step_result.done[0], step_result.max_step[0])
    
    def _multi_step(self, step_result):
        obs = step_result.obs[0]
        if len(obs.shape) == 2:
            result = []
            for obs_agent in obs:
                result.append(obs_agent)
            obs = result
        return (obs, list(step_result.reward), list(step_result.done), list(step_result.max_step))
    
    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result