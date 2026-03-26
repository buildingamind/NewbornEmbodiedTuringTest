"""
ports.py

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("localhost", port))
        except socket.error:
            return True
    return False


def claim_port(reserved: set, max_retries: int = 1000) -> int:
    """Picks a free port, records it in *reserved*, and returns it.

    Intended for use in the main process only. The caller maintains *reserved*
    as an in-process set of ports that have been handed out but not yet bound
    by their Unity executable. Checking *reserved* closes the TOCTOU window
    that exists when multiple workers each call random_port() independently:
    the main process never returns the same port twice until the previous holder
    has finished and removed it from *reserved*.

    Args:
        reserved: Mutable set owned by the caller. The chosen port is added
            before returning and should be discarded once the task completes.
        max_retries: Maximum candidate ports to try before raising.

    Raises:
        RuntimeError: If no available port is found after max_retries attempts.
    """
    for _ in range(max_retries):
        port = int(np.random.choice(LEGAL_PORTS))
        if port not in reserved and not _port_in_use(port):
            reserved.add(port)
            return port
    raise RuntimeError(f"Could not find an available port after {max_retries} attempts")


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
