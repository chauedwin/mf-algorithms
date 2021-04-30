#!/usr/bin/env python
# coding: utf-8

import numpy as np

class MF:

    '''
    Class for Matrix Factorization 
    
    A randomized iterative matrix factorization algorithm for matrix equations 
	of the form AS = X, where X is the data matrix and A, S are the factor matrices solved for.
    
    Parameters:
	-------------
	X:      array
            The data matrix "X" to be factored
	k:      int
            The factor dimension chosen 
	s1:     int, optional
            The block size of left factor matrix "A" used to update "S" (default is 1)
            Ignored in ALS update
	s2:     int, optional
            The block size of right factor matrix "S" used to update "A" (default is 1)
            Ignored in ALS update
    sub:    string, optional
            The subroutine used in the factorization updates (default is 'als')
            
            
    Methods:
    -------------
    solve(niter = 100, siter = 1, errseq = False)
        niter:  number of "alternating iterations"
        siter:  number of "subiterations" (iterations of subroutine)
        errseq: whether to store relative error of each iteration 
        
    get_error()
        returns the relative error of the resulting approximation 
	'''

    def __init__(self, X, k, **kwargs):
        self.X = X
        #initialize A and S if not provided
        self.A = kwargs.get('A', np.random.rand(np.shape(X)[0], k))
        self.S = kwargs.get('S', np.random.rand(k, np.shape(X)[1]))
        self.s1 = kwargs.get('s1', 1)
        self.s2 = kwargs.get('s2', 1)
        self.sub = kwargs.get('sub', 'als')
        
        if self.sub == "bgs":
            assert self.s1 <= k, "s1 should be less than k"
            assert self.s2 <= k, "s2 should be less than k"
        if self.sub == "qbrk" or self.sub == "brk":
            assert self.s1 <= X.shape[0], "s1 should be less than the number of rows"
            assert self.s2 <= X.shape[1], "s2 should be less than the number of columns"
            
            
    def solve(self, **kwargs):
        
        niter = kwargs.get('niter', 100)
        siter = kwargs.get('siter', 1)
        eps = kwargs.get('eps', 1e-3)
        errseq = kwargs.get('errseq', False)
        
        num_row, num_col = self.X.shape
        row_ind = np.arange(num_row)
        col_ind = np.arange(num_col)
        
        prop = num_row / num_col
        diff = 0
        
        if isinstance(errseq, (int, np.uint)):
            seq = list()
            
        for n in np.arange(niter):
            for p in np.arange(np.floor(prop + diff)):
                row = np.random.choice(row_ind, size = 1)
                if len(row_ind) == 1:
                    row_ind = np.arange(num_row)
                else:
                    row_ind = np.delete(row_ind, np.argwhere(row_ind == row))
                if self.sub == 'als':
                    self.A = self.leftals(self.X, self.A, self.S, row)
                if self.sub == 'brk':
                    self.A = self.leftbrk(self.X, self.A, self.S, row, s = self.s2, siter = siter, eps = eps)
                if self.sub == 'ubrk':
                    self.A = self.leftubrk(self.X, self.A, self.S, row, s = self.s2, siter = siter, eps = eps)
                if self.sub == 'bgs':
                    self.A = self.leftbgs(self.X, self.A, self.S, row, s = self.s2, siter = siter, eps = eps)
            
            diff = prop + diff - np.floor(prop + diff)
            
            col = np.random.choice(col_ind, size = 1)
            if len(col_ind) == 1:
                col_ind = np.arange(num_col)
            else:
                col_ind = np.delete(col_ind, np.argwhere(col_ind == col))
            
            if self.sub == 'als':
                self.S = self.leftals(self.X.T, self.S.T, self.A.T, row).T
            if self.sub == 'brk':
                self.S = self.leftbrk(self.X.T, self.S.T, self.A.T, row, s = self.s1, siter = siter, eps = eps).T
            if self.sub == 'ubrk':
                self.S = self.leftubrk(self.X.T, self.S.T, self.A.T, row, s = self.s1, siter = siter, eps = eps).T
            if self.sub == 'bgs':
                self.S = self.leftbgs(self.X.T, self.S.T, self.A.T, row, s = self.s1, siter = siter, eps = eps).T
                
            if errseq:
                if isinstance(errseq, (int, np.uint)) and ((n + 1) % errseq == 0 or n == 0):
                    seq.append(np.linalg.norm(self.X - self.A @ self.S) / np.linalg.norm(self.X))
            
        return seq 
        
      
    def get_error(self, **kwargs):
    
        '''
        returns the resulting relative error of the approximation
        '''
        
        return np.linalg.norm(self.X - self.A @ self.S) / np.linalg.norm(self.X)
            
            
    def weightsample(self, F, mode, **kwargs):
    
        '''
        computes the probability vector for a matrix mode for weighted sampling
        
        Parameters:
        ---------------
        F:      matrix from which we want to weight sample 
        mode:   either 1 or 0 (1 representing row and 0 representing column)
        '''
        
        prob = np.linalg.norm(F, axis = mode)
        return (prob / np.sum(prob))
        
        
    def leftals(self, X, lf, rf, row, **kwargs):
    
        '''
        Least squares update step 
        
        Parameters:
        ---------------
        X:      array
                The data matrix "X" to be factored
        lf:     array
                The left factor matrix to be updated
        rf:     array
                The right factor matrix used in the update
        row:    int 
                The row of the data matrix "X" used in the update
        '''
    
        siter = kwargs.get('siter', 1)
        for i in np.arange(siter):
            lf[row, :] = np.linalg.lstsq(rf.T, X[row, :].T, rcond = None)[0].T
        
        return lf
        
        
    def leftbrk(self, X, lf, rf, row, **kwargs):
    
        '''
        Block randomized Kaczmarz update step (the subset of columns in the right factor matrix are chosen via weighted sample)
        
        Parameters:
        ---------------
        X:      array
                The data matrix "X" to be factored
        lf:     array
                The left factor matrix to be updated
        rf:     array
                The right factor matrix used in the update
        row:    int 
                The row of the data matrix "X" used in the update
        '''
        
        siter = kwargs.get('siter', 1)
        eps = kwargs.get('eps', 1e-3)
        s = kwargs.get('s', 1)
        
        
        for i in np.arange(siter):
            if s == 1:
                # sample index for entry of data matrix
                kaczcol = np.random.choice(rf.shape[1], size = s, p = self.weightsample(rf, 0), replace = False)
                lf[row, :] = lf[row, :] + (X[row, kaczcol] - lf[row, :] @ rf[:, kaczcol]) / (np.linalg.norm(rf[:, kaczcol])**2) * rf[:, kaczcol].T 
            else:
                # sample until at least one row/column is nonzero
                resample = True
                while(resample):
                    kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)
                    if (np.linalg.norm(rf[:, kaczcol]) > 0):
                        resample = False

                # compute BRK step
                lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[:, kaczcol].T, (X[None, row, kaczcol] - lf[row, :] @ rf[:, kaczcol]).T, rcond = None)[0].T	
            if np.linalg.norm(lf[row, :] @ rf - X[row, :]) < eps:
                break
        return lf
        
        
        
    def leftubrk(self, X, lf, rf, row, **kwargs):
    
        '''
        Uniform Block randomized Kaczmarz update step (the subset of columns in the right factor matrix are uniformly sampled)
        
        Parameters:
        ---------------
        X:      array
                The data matrix "X" to be factored
        lf:     array
                The left factor matrix to be updated
        rf:     array
                The right factor matrix used in the update
        row:    int 
                The row of the data matrix "X" used in the update
        '''
        
        siter = kwargs.get('siter', 1)
        eps = kwargs.get('eps', 1e-3)
        s = kwargs.get('s', 1)
        
        for i in np.arange(siter):
            kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)
            if s == 1:
                lf[row, :] = lf[row, :] + (X[row, kaczcol] - lf[row, :] @ rf[:, kaczcol]) / (np.linalg.norm(rf[:, kaczcol])**2) * rf[:, kaczcol].T 
                if (i % 100 == 0):
                    if np.linalg.norm(lf[row, :] @ rf - X[row, :]) < eps:
                        #print("left break " + str(i))
                        break
            else:
                lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[:, kaczcol].T, (X[None, row, kaczcol] - lf[row, :] @ rf[:, kaczcol]).T, rcond = None)[0].T
                if np.linalg.norm(lf[row, :] @ rf - X[row, :]) < eps:
                    #print("left break " + str(i))
                    break
                
        return(lf)
        
        
    def leftbgs(self, X, lf, rf, row, **kwargs):
    
        '''
        Block Gauss-Seidel update rule (the subset of columns in the right factor matrix are chosen via weighted sample)
        
        Parameters:
        ---------------
        X:      array
                The data matrix "X" to be factored
        lf:     array
                The left factor matrix to be updated
        rf:     array
                The right factor matrix used in the update
        row:    int 
                The row of the data matrix "X" used in the update
        '''
        
        siter = kwargs.get('siter', 1)
        eps = kwargs.get('eps', 1e-3)
        s = kwargs.get('s', 1)
        k = lf.shape[1]
        
        for j in np.arange(siter):
            if s2 == 1:
                gsrow = np.random.choice(rf.shape[0], size = s, p = self.weightsample(rf, 1), replace = False)
            else:
                resample = True
                while(resample):
                    gsrow = np.random.choice(rf.shape[0], size = s, replace = False)           
                    if (np.linalg.norm(rf[gsrow, :] > 0)):
                        resample = False
                # compute BGS step
                lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[gsrow, :].T, (X[row, :] - lf[row, :] @ rf).T, rcond = None)[0].T @ np.eye(k)[gsrow, :]
            if np.linalg.norm(lf[row, :] @ rf - X[row, :]) < eps:
                break
        return(lf)
        
    