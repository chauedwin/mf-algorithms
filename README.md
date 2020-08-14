# MF Algorithms

[![PyPI Version](https://img.shields.io/pypi/v/mf-algorithms.svg)](https://pypi.org/project/mf-algorithms/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/mf-algorithms.svg)](https://pypi.org/project/mf-algorithms/)

MF Algorithms contains various matrix factorization methods utilizing different iterative update rules.

---

## Installation

To install MF Algorithms, run this command in your terminal:

```bash
$ pip install -U mf-algorithms
```

This is the preferred method to install MF Algorithms, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Usage
First import `functions` from the package. `scipy.sparse` is also useful for creating toy sparse matrices to test the algorithms, thought we will manually generate factor matrices and multiply them to guarantee its rank.

```python
>>> import numpy as np
>>> from mf_algorithms import functions
```

### Matrix Factorization

```python
>>> dim1 = 1000
>>> dim2 = 1000
>>> k = 50
>>> factors = np.random.choice(4, size=(dim1,k), p=np.array([0.97, 0.01, 0.01, 0.01]))
>>> weights = np.random.choice(2, size=(k, dim2), p=np.array([0.999, 0.001]))
>>> mat = factors @ weights
>>> A, S, error = functions.mf(data = mat, k = 50, s1 = 1, s2 = 1, niter = 100, siter = 1, update = 'als', errseq = False)
```

## Citing
If you use our work in an academic setting, please cite our paper:



## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. Your day-to-day work should exist on branches separate from `master`. It is recommended to commit to development branches and make pull requests to master.4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes each set of changes to `master`
atomic and as a side effect naturally encourages small well defined PR's.


#### Additional Optional Setup Steps:
* Create an initial release to test.PyPI and PyPI.
    * Follow [This PyPA tutorial](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives), starting from the "Generating distribution archives" section.

* Create a blank github repository (without a README or .gitignore) and push the code to it.

* Delete these setup instructions from `README.md` when you are finished with them.
