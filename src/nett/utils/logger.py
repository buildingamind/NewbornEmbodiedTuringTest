"""
Logger

Classes:
    Logger
"""

#Packages for making the environment
import uuid #needed for the communicator
import os #Files and directories

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

# Create the StringLogChannel class. 
class Logger(SideChannel):
    """
    This class is used to log information from the environment to a file. This is how logging info is communicated between python and unity. It is a subclass of the SideChannel class from the mlagents_envs package.

    Methods:
        on_message_received: Method from Sidechannel interface. This method gets a message from unity and writes it to the log file.
        send_string: Method from Sidechannel interface. This method send a message to unity.
        log_str: This method is used to log a string to the file.
    """
    def __init__(self, log_title, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7")) # TODO why this UUID?
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir, f"{log_title}.csv")
        self.f = open(f_name, "w")

    def on_message_received(self, msg: IncomingMessage) -> None:
        """Method from Sidechannel interface. This method gets a message from unity and writes it to the log file."""
        self.f.write(msg.read_string()) #Write message to log file
        self.f.write("\n") #add new line character

    #This is here because it is required and I currently don't use it.
    def send_string(self, data: str) -> None:
        """Method from Sidechannel interface. This method send a message to unity."""
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def log_str(self, msg: str) -> None:
        """This method is used to log a string to the file."""
        self.f.write(msg)
        self.f.write("\n")

    def __del__(self) -> None:
        """This is called when the environment is shut down. It closes the log file."""
        self.f.close()
