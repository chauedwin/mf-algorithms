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
>>> from mf_algorithms import functions
```

### Matrix Factorization

```python
>>> test, factor, weight = functions.createmat(dim = 200, k = 4, s = 1)
>>> A, S, error = functions.mf(data = test, k = 2, s = 1, niter = 100, siter = 1, solver = 'als', errseq = False, reinit = 1)
```
We can create a sparse matrix with 50% density using a custom function called `createmat`, which generates a square matrix of dimension `dim` with rank `k` using a seed `s`, and returns the square matrix along with the two factor matrices.

Then factorize the data matrix with `functions.mf`. The parameters are as follows:
* `data` takes a data matrix (matrix)
* `k` takes the factor dimension (int)
* `s` takes the number of rows/columns to sample per iterative solving step (int)
* `niter` takes the number of left and right factor updates (int)
* `siter` takes the number of iterative steps per left/right factor update (int)
* `solver` takes the type of iterative solver(ALS, BRK, or BGS) (string)
* `errseq` returns the entire sequence of relative errors if `True` and only the last relative error if `False` (bool)
* `reinit` reruns the factorization the specified number of times and returns the initialization with the lowest relative error (int)

It then returns the following:
* left factor matrix
* right factor matrix
* error(`array` if `errseq` was true, `float` otherwise)

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
