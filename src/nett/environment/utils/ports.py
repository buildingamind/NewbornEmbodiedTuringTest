"""
logger.py

Functions:
    _port_in_use(port: int) -> bool
    random_port() -> int
"""

import socket
import numpy as np

# range of all of the ports that can are user accessible
LEGAL_PORTS = np.arange(1024, 49151)


def _port_in_use(port) -> bool:
    """This function checks if a port is in use. It returns True if the port is in use and False if it is not."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    finally:
        sock.close()
    return False


def random_port(max_retries: int = 1000) -> int:
    """Returns a random port that is not in use.

    Args:
        max_retries: Maximum number of attempts before raising RuntimeError.

    Raises:
        RuntimeError: If no available port is found after max_retries attempts.
    """
    for _ in range(max_retries):
        port = int(np.random.choice(LEGAL_PORTS))
        if not _port_in_use(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_retries} attempts")
