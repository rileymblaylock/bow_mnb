# run python -m tests.test_module from root directory

import unittest
from src.bow_mnb.mnb import *

#Test
train_and_test('./src/bow_mnb/data/train/', validationPath='./src/bow_mnb/data/validation/', testPath='./src/bow_mnb/data/test/test/', fileType='res')