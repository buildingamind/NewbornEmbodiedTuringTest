#Packages for opening unity package
import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import socket
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid
import os
import hydra
from omegaconf import DictConfig, OmegaConf

test_eps = 1200
ep_steps = 1000
test_steps = ep_steps * test_eps
rest_steps = ep_steps * 100#test_eps
env_path = "../../Env/rearing_chamber"

def port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    return False

# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self, run_name, mode, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir,run_name+"_"+mode+".csv")
        self.f = open(f_name, 'w')

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        self.f.write(msg.read_string())
        self.f.write("\n")

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)
    
    def custom_message(self, msg: str) -> None:
        self.f.write(msg)
        self.f.write("\n")

    def __del__(self):
        self.f.close()

class ChickEnv(gym.Wrapper):
    def __init__(self, run_id: str, useShip=False, sideView=False, mode="rest", path=None, base_port=5004):
        args = []
        if config.useShip: args.extend(["--use-ship", "true"])
        if config.sideView: args.extend(["--side-view", "true"])
        if config.mode == "exp1": 
            args.extend(["--test-mode","true"])
        elif config.mode == "exp2":
            args.extend(["--test-mode" ,"true"])
            args.extend(["--experiment-2","true"])
        elif config.mode != "rest":
            print("Running in rest mode, mode must be in [exp1,exp2,rest]")
            
        while port_in_use(base_port):
            base_port += 1

        #Creat unity environment and its communicator
        self.string_log = StringLogChannel(config.runID, mode=config.mode)
        env = UnityEnvironment(config.path, side_channels=[self.string_log], 
                additional_args=args, base_port=base_port)
        self.env = UnityToGymWrapper(env,uint8_visual=True)
        super().__init__(self.env)
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info
    
    def log(self, msg: str) -> None:
        self.string_log.custom_message(msg)
