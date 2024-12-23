# Copyright 2022 The Wood Lab, Indiana University Bloomington. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Initialize the NETT library
"""

import os
import logging

# release version
from ._version import __version__

# simplify imports
from .nett import NETT
from .body.wrappers import list_wrappers
from .brain.params import list_algorithms, list_encoders, list_policies, list_rewards

# set up logging
logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger("nett")

# Alter permissions for ml-agents binaries which are shared between users
for tmp_dir in [
    "/tmp/ml-agents-binaries",
    "/tmp/ml-agents-binaries/binaries",
    "/tmp/ml-agents-binaries/tmp",
]:
    # check if directory is correct permission
    if os.stat(tmp_dir).st_mode % 0o1000 != 0o777:
        # check if directory is owned by user
        if os.stat(tmp_dir).st_uid == os.getuid():
            # change permission of directory
            os.chmod(tmp_dir, 0o1777)
        else:
            logger.error(
                f"Error: '{tmp_dir}' does not have correct permissions and cannot be changed. If you have superuser access, you can run the following command to change the permissions: 'sudo chmod 1777 {tmp_dir}'. Otherwise, request {os.stat(tmp_dir).st_uid} to run 'chmod 1777 {tmp_dir}'."
            )
