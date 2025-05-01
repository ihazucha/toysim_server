import os
from enum import Enum


class Environment(Enum):
    VEHICLE = "vehicle"
    SIM = "sim"

DEBUG = os.getenv("DEBUG")
_ENV = os.getenv("ENV")
ENV = Environment(_ENV) if _ENV else None
DEFAULT_ENV = Environment.SIM

if ENV is None:
    ENV = DEFAULT_ENV

# ANSI escape codes for colors
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_RESET = "\033[0m"

def pwarn(text: str, *args, **kwargs):
    print(f"{COLOR_YELLOW}[W]{COLOR_RESET} {text}", *args, **kwargs)

def perror(text: str, *args, **kwargs):
    print(f"{COLOR_RED}[E]{COLOR_RESET} {text}", *args, **kwargs)

def pinfo(text: str, *args, **kwargs):
    print(f"{COLOR_BLUE}[I]{COLOR_RESET} {text}", *args, **kwargs)

def pdebug(text: str, *args, **kwargs):
    if DEBUG:
        print(f"{COLOR_MAGENTA}[D]{COLOR_RESET} {text}", *args, **kwargs)