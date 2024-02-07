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
import logging
from pathlib import Path
# simplify imports
from nett.brain.builder import Brain
from nett.body.builder import Body
from nett.environment.builder import Environment
from nett.nett import NETT

# release version
__version__ = "0.1"

# change permissions of the ml-agents binaries directory
os.chmod('/tmp/ml-agents-binaries', 0o1777)
os.chmod('/tmp/ml-agents-binaries/binaries', 0o1777)
os.chmod('/tmp/ml-agents-binaries/bin', 0o1777)

# path to store library cache (such as configs etc)
cache_dir = Path.joinpath(Path.home(), ".cache", "nett")

# set up logging
logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
