"""
Initializes the environment module.
"""
# NOTE: Import was causing circular import error (nett.environment -> environment)
# simplify imports
from nett.environment.configs import list_configs
from nett.environment.configs import IdentityAndView, Parsing, Slowness, Smoothness, Binding, OneShotViewInvariant
