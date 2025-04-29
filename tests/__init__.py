# tests/__init__.py

# This file marks the tests directory as a Python package.
# It can be used to define package-wide fixtures, imports, or configurations for testing.

__all__ = [
    'test_enhancement',
]

# Import test modules to make them available when importing the tests package
from .test_enhancement import test_processing

test_enhancement = [
    'test_processing',
]
# Add any other test modules here
# For example:
# from .test_model import test_model
# from .test_utils import test_utils
# test_enhancement.append('test_model')
# test_enhancement.append('test_utils')
# __all__ = test_enhancement
# This allows you to import all tests using:
# from tests import *
# Or you can import specific tests:
# from tests import test_processing
# from tests import test_model
# You can also define fixtures or setup code here if needed
# For example, if you want to set up a temporary directory for tests:
import os
import tempfile
import pytest
import shutil
from typing import Optional
import numpy as np
import torch    