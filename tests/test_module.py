# run python -m tests.test_module from root directory

import unittest
from src.simple_mnb.simple_mnb import *

#Test
train_and_test('./src/simple_mnb/data/train/', validationPath='./src/simple_mnb/data/validation/')