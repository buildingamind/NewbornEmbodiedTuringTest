"""
    This file defines all methods that are used to check that user-defined parameters are of the approriate data type
"""
from nett import Brain, Body, Environment, NETT

def check_params(cls = object, params = {}):
    # if cls.__instancecheck__(Brain)