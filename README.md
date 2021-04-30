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
First import `functions` from the package. `scipy.sparse` is also useful for creating toy sparse matrices to test the algorithms, though we will manually generate factor matrices and multiply them to guarantee its rank.

```python
>>> import numpy as np
>>> from mf_algorithms import mf
```

### Matrix Factorization

```python
>>> dim1 = 1000
>>> dim2 = 1000
>>> k = 50
>>> A = np.random.choice(4, size=(dim1,k), p=np.array([0.97, 0.01, 0.01, 0.01]))
>>> S = np.random.choice(2, size=(k, dim2), p=np.array([0.999, 0.001]))
>>> X = A @ S
>>> model = MF(X, k, sub = 'ubrk', s1 = 200, s2 = 200)
>>> model.solve(niter = 1000)
```

## Citing
If you use our work in an academic setting, please cite our paper:

Edwin Chau and Jamie Haddock. “On Application of Block Kaczmarz Methods in Matrix Factorization.” Submitted. https://arxiv.org/abs/2010.10635, 2020.

## Authors

- Edwin Chau
- Jamie Haddock