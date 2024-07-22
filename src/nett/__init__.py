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

'''
Initialize the NETT library
'''

import os
import stat
import logging
from pathlib import Path
# simplify imports
from nett.brain.builder import Brain
from nett.body.builder import Body
from nett.environment.builder import Environment
from nett.nett import NETT

from nett.brain import list_encoders, list_algorithms, list_policies
from nett.environment import list_configs

# release version
__version__ = "0.3.1"

# change permissions of the ml-agents binaries directory

# path to store library cache (such as configs etc)
cache_dir = Path.joinpath(Path.home(), ".cache", "nett")

# set up logging
logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# path to store ml-agents binaries
for tmp_dir in ["/tmp/ml-agents-binaries", "/tmp/ml-agents-binaries/binaries", "/tmp/ml-agents-binaries/tmp"]:
  # TODO: May need to allow for permissions other than X777
  if stat.S_IMODE(os.stat(tmp_dir).st_mode) % 0o1000 != 0o777:
    # TODO: May need to check for permissions other than W_OK
    if os.stat(tmp_dir).st_uid == os.getuid() or os.access(tmp_dir, os.W_OK):
      os.chmod(tmp_dir, 0o1777)
    else:
      logger.warning(f"You do not have permission to change the necessary files in '{tmp_dir}'.")