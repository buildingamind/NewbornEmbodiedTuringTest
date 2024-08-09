"""
environment.py

Classes:
    Logger(SideChannel)

Functions:
    port_in_use(port: int) -> bool
"""

import uuid
import os
import socket
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

# Create the StringLogChannel class. This is how logging info is communicated between python and unity
class Logger(SideChannel):
    """
    This class is used to log information from the environment to a file. It inherits from the SideChannel class
    
    Methods:
        on_message_received(msg: IncomingMessage) -> None
        send_string(data: str) -> None
        log_str(msg: str) -> None
        __del__() -> None
    """
    def __init__(self, log_title, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir, f"{log_title}.csv")
        self.f = open(f_name, "w")

    #Method from Sidechannel interface.
    def on_message_received(self, msg: IncomingMessage) -> None:
        """This method is called when a message is received from unity."""
        self.f.write(msg.read_string()) #Write message to log file
        self.f.write("\n") #add new line character

    #This is here because it is required and I currently don"t use it.
    def send_string(self, data: str) -> None:
        """Method from Sidechannel interface. This method send a message to unity."""
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def log_str(self, msg: str) -> None:
        """This method writes a custom string to the log file"""
        self.f.write(msg)
        self.f.write("\n")

    def __del__(self) -> None:
        """This is called when the environment is shut down"""
        self.f.close()
