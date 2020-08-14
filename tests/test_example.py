import numpy as np
from mf_algorithms import functions

def test_sample():
	mat = np.random.choice(10, size=(10, 10))
	prob0 = functions.weightsample(mat, 0)
	prob1 = functions.weightsample(mat, 1)
	assert round(sum(prob0), 5) == 1
	assert round(sum(prob1), 5) == 1
	