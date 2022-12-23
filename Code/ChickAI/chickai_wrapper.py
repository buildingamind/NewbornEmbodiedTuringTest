#Packages for making the environment
import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid #needed for the communicator
import socket #Manging ports
import os #Files and directories

#Hack for making it easy to run multiple environments
def port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    return False

# Create the StringLogChannel class. This is how logging info is communicated between python and unity
class Logger(SideChannel):
    def __init__(self, log_title, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir,log_title + ".csv")
        self.f = open(f_name, 'w')

    #Method from Sidechannel interface. This method gets a message from unity
    def on_message_received(self, msg: IncomingMessage) -> None:
        self.f.write(msg.read_string()) #Write message to log file
        self.f.write("\n") #add new line character

    #Method from Sidechannel interface. This method send a message to unity.
    #This is here because it is required and I currently don't use it.
    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)
    
    #This method writes a custom string to the log file
    def log_str(self, msg: str) -> None:
        self.f.write(msg)
        self.f.write("\n")

    #This is called when the environment is shut down
    def __del__(self):
        self.f.close()

#Gym wrapper for the viewpoint environment. New gym wrapper (specifically for argument parsing) needs to be made for
#a different experiment setup.
class ViewpointEnv(gym.Wrapper):
    def __init__(self, run_id: str, use_ship=False, side_view=False, mode="rest", log_path="./Env_Logs/", env_path=None, base_port=5004, **kwargs):
        #Parse arguments and determine which version of the environment to use.
        args = []
        if use_ship: args.extend(["--use-ship", "true"])
        if side_view: args.extend(["--side-view", "true"])
        self.mode = mode
        if mode == "exp1": 
            args.extend(["--test-mode","true"])
        elif mode == "exp2":
            args.extend(["--test-mode" ,"true"])
            args.extend(["--experiment-2","true"])
        elif mode != "rest":
            print("Running in rest (imprint) mode, mode must be in [exp1,exp2,rest]")
            self.mode = "rest"

        #Find unused port 
        while port_in_use(base_port):
            base_port += 1

        #Create logger
        self.log = Logger(run_id, log_dir=log_path)

        #Create environment and connect it to logger
        env = UnityEnvironment(env_path, side_channels=[self.log], additional_args=args, base_port=base_port)
        self.env = UnityToGymWrapper(env, uint8_visual=True)
        super().__init__(self.env)
        
    #Step the environment for one timestep
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    
    #Write to the log file
    def log(self, msg: str) -> None:
        self.log.log_str(msg)
    
    #Close environment
    def close(self):
        self.env.close()
        del self.log
    
    #This function is needed since episode lengths and the number of stimuli are determined in unity
    def steps_from_eps(self, eps):
        step_per_episode = 1000
        numb_conditions = 12
        if self.mode == "rest":
            return step_per_episode * eps
        else:
            return step_per_episode * eps * numb_conditions
        
    def reset():
        self.env.reset()
