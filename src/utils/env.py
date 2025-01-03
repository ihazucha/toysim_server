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

# -----------------------------------------------


def pdebug(msg: str):
    if DEBUG:
        pdebug(msg)
