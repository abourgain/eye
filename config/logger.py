"""
Create a logger to log the events in the application
"""

import logging
import os
from os import path

SCRIPT_PATH = path.dirname(path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, os.pardir))

LOGS_PATH = path.join(ROOT_PATH, "logs")
if not path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH, exist_ok=True)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(path.join(LOGS_PATH, "logs.log"))
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)