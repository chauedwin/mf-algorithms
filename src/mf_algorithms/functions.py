#!/usr/bin/env python
# coding: utf-8


import numpy as np
import multiprocessing as mp
import random
import re
import glob
import sys

def weightsample(data, mode):
	''' computes the row-wise or column-wise probability vector for a matrix,
	useful for a weighted sample of rows or columns
	'''
	# mode 1 samples rows
	prob = np.linalg.norm(data, axis=mode)
	return(prob / sum(prob))


def leftals(data, s2, lf, rf, siter, row, eps):
	''' Left ALS update
	Solves and updates x in the system Ax = b using least squares. 
	'''
	
	# perform linear reg update 
	for i in np.arange(int(siter)):
		lf[row, :] = np.linalg.lstsq(rf.T, data[row, :].T, rcond = None)[0].T
		#lf[row, :] = np.linalg.solve(np.matmul(rf, rf.T), np.matmul(rf, data[row, :].T)).T
	return(lf)


def rightals(data, s1, lf, rf, siter, col, eps):
	''' Right ALS update
	Solves and updates x in the system xA = b using least squares.
	Equivalent to using the left update to update x.T in system A.Tx.T = b.T
	'''
	
	# perform linear reg update 
	for i in np.arange(siter):
		rf[:, col] = np.linalg.lstsq(lf, data[:, col], rcond = None)[0]
		#rf[:, col] = np.linalg.solve(np.matmul(lf.T, lf), np.matmul(lf.T, data[:, col]))
	return(rf)


def leftbrk(data, s2, lf, rf, siter, row, eps):
	''' Left BRK update
	Solves and updates x in the system Ax = b using Block Randomized Kaczmarz.
	The Kaczmarz columns are selected through WEIGHTED sampling.
	'''
	
	for i in np.arange(siter):
		if s2 == 1:
			# sample index for entry of data matrix
			kaczcol = np.random.choice(rf.shape[1], size = s2, p = weightsample(rf, 0), replace = False)
			lf[row, :] = lf[row, :] + (data[row, kaczcol] - lf[row, :] @ rf[:, kaczcol]) / (np.linalg.norm(rf[:, kaczcol])**2) * rf[:, kaczcol].T 
		else:
			# sample s.t. at least one row/column is nonzero
			resample = True
			while(resample):
				kaczcol = np.random.choice(rf.shape[1], size = s2, replace = False)
				if (np.linalg.norm(rf[:, kaczcol]) > 0):
					resample = False

			# compute BRK step
			lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[:, kaczcol].T, (data[None, row, kaczcol] - lf[row, :] @ rf[:, kaczcol]).T, rcond = None)[0].T	
		if np.linalg.norm(lf[row, :] @ rf - data[row, :]) < eps:
			break
	return(lf)


def rightbrk(data, s1, lf, rf, siter, col, eps):
	''' Right BRK update
	Solves and updates x in the system xA = b using Block Randomized Kaczmarz.
	The Kaczmarz columns are selected through WEIGHTED sampling.
	Equivalent to using the left update to update x.T in system A.Tx.T = b.T
	'''
	
	for i in np.arange(siter):
		if s1 == 1:
			# sample index for entry of data matrix
			kaczrow = np.random.choice(lf.shape[0], size = s1, p = weightsample(lf, 1), replace = False)
			rf[:, col] = rf[:, col] + (data[kaczrow, col] - lf[kaczrow, :] @ rf[:, col]) / (np.linalg.norm(lf[kaczrow, :])**2) * lf[kaczrow, :].T
		else:
			# sample s.t. at least one row/column is nonzero
			resample = True
			while(resample):
				kaczrow = np.random.choice(lf.shape[0], size = s1, replace = False)
				if (np.linalg.norm(lf[kaczrow]) > 0):
					resample = False

			# compute BRK step
			rf[:, col] = rf[:, col] + np.linalg.lstsq(lf[kaczrow, :], (data[kaczrow, col, None] - lf[kaczrow, :] @ rf[:, col]), rcond = None)[0]
		if np.linalg.norm(lf @ rf[:, col] - data[:, col]) < eps:
			break
	return(rf)


def leftqbrk(data, s2, lf, rf, siter, row, eps):
	''' Left QBRK update
	Solves and updates x in the system Ax = b using Block Randomized Kaczmarz.
	The Kaczmarz columns are selected through UNIFORM sampling.
	'''
	
	for i in np.arange(siter):
		kaczcol = np.random.choice(rf.shape[1], size = s2, replace = False)
		if s2 == 1:
			lf[row, :] = lf[row, :] + (data[row, kaczcol] - lf[row, :] @ rf[:, kaczcol]) / (np.linalg.norm(rf[:, kaczcol])**2) * rf[:, kaczcol].T 
			if (i % 100 == 0):
				if np.linalg.norm(lf[row, :] @ rf - data[row, :]) < eps:
					#print("left break " + str(i))
					break
		else:
			lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[:, kaczcol].T, (data[None, row, kaczcol] - lf[row, :] @ rf[:, kaczcol]).T, rcond = None)[0].T
			if np.linalg.norm(lf[row, :] @ rf - data[row, :]) < eps:
				#print("left break " + str(i))
				break
			
	return(lf)


def rightqbrk(data, s1, lf, rf, siter, col, eps):
	''' Right QBRK update
	Solves and updates x in the system xA = b using Block Randomized Kaczmarz.
	The Kaczmarz columns are selected through UNIFORM sampling.
	Equivalent to using the left update to update x.T in system A.Tx.T = b.T
	'''
	
	for i in np.arange(siter):
		kaczrow = np.random.choice(lf.shape[0], size = s1, replace = False)
		if s1 == 1:
			rf[:, col] = rf[:, col] + (data[kaczrow, col] - lf[kaczrow, :] @ rf[:, col]) / (np.linalg.norm(lf[kaczrow, :])**2) * lf[kaczrow, :].T
			if (i % 100 == 0):
				if np.linalg.norm(lf @ rf[:, col] - data[:, col]) < eps:
					#print("right break " + str(i))
					break
		else:
			rf[:, col] = rf[:, col] + np.linalg.lstsq(lf[kaczrow, :], (data[kaczrow, col, None] - lf[kaczrow, :] @ rf[:, col]), rcond = None)[0]
			if np.linalg.norm(lf @ rf[:, col] - data[:, col]) < eps:
				#print("right break " + str(i))
				break
	return(rf)


def leftbgs(data, s2, lf, rf, siter, row, eps):
	''' Left BGS update
	Solves and updates x in the system Ax = b using Block Gauss-Seidel.
	The Gauss-Seidel rows are selected through WEIGHTED sampling.
	'''
	k = lf.shape[1]
	# inner loop for number of GS iterations
	for j in np.arange(siter):
		if s2 == 1:
			gsrow = np.random.choice(rf.shape[0], size = s2, p = weightsample(rf, 1), replace = False)
		else:
			resample = True
			while(resample):
				gsrow = np.random.choice(rf.shape[0], size = s2, replace = False)           
				if (np.linalg.norm(rf[gsrow, :] > 0)):
					resample = False
			# compute BGS step
			lf[row, :] = lf[row, :] + np.linalg.lstsq(rf[gsrow, :].T, (data[row, :] - lf[row, :] @ rf).T, rcond = None)[0].T @ np.eye(k)[gsrow, :]
		if np.linalg.norm(lf[row, :] @ rf - data[row, :]) < eps:
			break
	return(lf)


def rightbgs(data, s1, lf, rf, siter, col, eps):
	''' Right QBRK update
	Solves and updates x in the system xA = b using Block Gauss-Seidel.
	The Gauss-Seidel columns are selected through WEIGHTED sampling.
	Equivalent to using the left update to update x.T in system A.Tx.T = b.T
	'''
	
	k = lf.shape[1]
		# inner loop for number of GS iterations
	for j in np.arange(siter):
		if s1 == 1:
			gscol = np.random.choice(lf.shape[1], size = s1, p = weightsample(lf, 0), replace = False)
		else:
			resample = True
			while(resample):
				gscol = np.random.choice(lf.shape[1], size = s1, replace = False)
				if (np.linalg.norm(lf[:, gscol] > 0)):
					resample = False
			# compute BGS step
			rf[:, col] = rf[:, col] + np.eye(k)[:, gscol] @ np.linalg.lstsq(lf[:, gscol], (data[:, col] - lf @ rf[:, col]), rcond = None)[0]
		if np.linalg.norm(lf @ rf[:, col] - data[:, col]) < eps:
			break
	return(rf)


def solver(data, s1, s2, lf, rf, niter, siter, update, errseq, eps):
	
	if update == "als":
		leftupdate = leftals
		#rightupdate = rightals
	if update == "brk":
		leftupdate = leftbrk
		#rightupdate = rightbrk
	if update == "bgs":
		leftupdate = leftbgs
		#rightupdate = rightbgs
	if update == "qbrk":
		leftupdate = leftqbrk
		#rightupdate = rightqbrk
        
	r, c = data.shape
	prop = r / c
	diff = 0
	seqerr = list()

	r_dim, c_dim = data.shape
	r_ind = np.arange(r_dim)
	c_ind = np.arange(c_dim)
	rows = list()
	cols = list()
    
	for n in np.arange(niter):
		for p in np.arange(np.floor(prop + diff)):
			row = np.random.choice(r_ind, size = 1)
			rows.append(row)
			if(len(r_ind) == 1):
				r_ind = np.arange(r_dim)
			else:
				r_ind = np.delete(r_ind, np.argwhere(r_ind==row))
			lf = leftupdate(data, s2, lf, rf, siter, row, eps)
		diff = prop + diff - np.floor(prop + diff)
        
		col = np.random.choice(c_ind, size = 1)
		cols.append(col)
		if(len(c_ind) == 1):
			c_ind = np.arange(c_dim)
		else:
			c_ind = np.delete(c_ind, np.argwhere(c_ind==col))
        
		#rf = rightupdate(data, s1, lf, rf, siter, col, eps)
		rf = leftupdate(data.T, s1, rf.T, lf.T, siter, col, eps).T
            
		if (errseq > 0 and ((n + 1) % errseq == 0 or n == 0)):
			seqerr.append(np.linalg.norm(data - np.matmul(lf, rf)) / np.linalg.norm(data))
		
	return(lf, rf, seqerr)



def mf(data, k, s1 = 1, s2 = 0, niter = 100, siter = 1, update = 'als', errseq = 0, mult = 0.5, eps = 1e-3, reinit = 1):
    
	''' Matrix Factorization
	
	A randomized iterative matrix factorization algorithm for matrix equations 
	of the form AS = X.
	
	Parameters:
	-------------
	data: ndarray
		The data matrix "X" to be factored
	k: int
		The factor dimension chosen 
	s1: int
		The block size of left factor matrix "A" used to update "S"
		Ignored in ALS update
	s2: int
		The block size of right factor matrix "S" used to update "A"
		Ignored in ALS update
	niter: int
		The number of alternating iterations 
	siter: int
		The number of subiterations
	update: string from the set {"als", "brk", "qbrk", "bgs"}
		The type of matrix update. QBRK is the same as UBRK.
	errseq: int
		Calculates and returns the relative error of the factorization at multiples of 
		the errseq value starting at 0
		Default is 0, returning the final relative error
	mult: float
		Number to multiply the starting initializations of factor matrices
	eps: float
		Precision of brk/qbrk/bgs updates when siter > 1
	reinit: int
		Number of times to factorize the data matrix, factorization with 
		lowest final relative error will be returned
	'''
	
	if s2 == 0:
		s2 = s1

	if (data.shape[0] < data.shape[1]):
		data = data.T
		
    # make sure s is valid
	if update == "bgs":
		assert s1 <= k, "s1 should be less than k"
		assert s2 <= k, "s2 should be less than k"
	if update == "qbrk" or update == "brk":
		assert s1 <= data.shape[0], "s1 should be less than the number of rows"
		assert s2 <= data.shape[1], "s2 should be less than the number of columns"
	
    # set to negative 1 so we can guarantee an update for the first init
	finalerr = -1
    
	for l in np.arange(reinit):
		seqerr = list()
        
		# randomly initialize the factor matrices
		lfactor = np.random.rand(data.shape[0], k) * mult
		rfactor = np.random.rand(k, data.shape[1]) * mult
		#start_err = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
		#prev_err = start_err

		# outer loop for number of iterations 
		lfactor, rfactor, seqerr = solver(data, s1, s2, lfactor, rfactor, niter, siter, update, errseq, eps)

		# calculate ending error if no sequence needed
		if (errseq == 0):
			seqerr.append(np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data))

		# update after first init
		if (finalerr == -1):
			finalerr = seqerr
			lbest = lfactor
			rbest = rfactor
        # if not first, only update if final error is lower than overall best
		elif (finalerr[-1] > seqerr[-1]):
			finalerr = seqerr
			lbest = lfactor
			rbest = rfactor
	return(lbest, rbest, finalerr)


def read(filename): 
    with open(filename, 'r') as f:
        l = f.read().split(',')
    return(l)


def extracterr(tag, errfiles, titletag): 
    #r = re.compile(".*(" + tag + ").*")
    r = re.compile(tag)
    files = list(filter(r.match, errfiles))
    title = list()
    meanerr = list()
    stderr = list()
    for f in reversed(files):
        title.append("".join(re.findall(titletag + '([0-9]+k*)', f)[0]))
        meanerr.append(np.mean(np.asarray(read(f)[:-1]).astype(float)))
        stderr.append(np.std(np.asarray(read(f)[:-1]).astype(float)))
    return(title, meanerr, stderr)

	
def mfwrite(data, k, s1, s2, niter, siter, update, errseq, mult, q):
    A, S, error = mf(data, k, s1, s2, niter, siter, update, errseq, mult)
    q.put(error[0])


def mpmf(data, k, s1, s2, niter, siter, update, mult, filename, loop, cores = mp.cpu_count()):
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(mfwrite, (data, k, s1, s2, niter, siter, update, 0, mult, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


def listener(q, textfile):
	'''listens for messages on the q, writes to file. '''
	with open(textfile, 'w') as f:
		while 1:
			m = q.get()
			if m == 'kill':
				f.write('killed')
				break
			f.write(str(m) + ', ')
			f.flush()
	

def alsupdate(data, lf, rf, s, siter):
    """
    Alternating Least Squares Update
    
    It computes a least squares solution for each update. Computationally expensive(depending on the matrix shapes) but exact.
    """
    row = np.random.randint(data.shape[0], size = 1)
    col = np.random.randint(data.shape[1], size = 1)
            
    # perform linear reg update 
    for i in np.arange(siter):
        rf[:, col] = np.linalg.solve(np.matmul(lf.T, lf), np.matmul(lf.T, data[:, col]))
        lf[row, :] = np.linalg.solve(np.matmul(rf, rf.T), np.matmul(rf, data[row, :].T)).T
        #rf[:, col] = np.linalg.solve(lf, data[:,col])
        #lf[row, :] = np.linalg.solve(rf.T, data[row, :].T).T
    return(lf, rf)


def brkupdate(data, lf, rf, s, siter):
    """
    Block Randomized Kaczmarz Update
    
    It approximates the least squares solution at each update. It performs a weighted sampling to choose
    a row/col of the lf/rf matrix to update. If the block size is 1, it performs a normal RK update, otherwise
    it performs a BRK update. The blocks are sampled such that there is at least one nonzero row/col.
    """
            
    # weighted sampling of row and column from data matrix
    # specifying size returns an array rather than a scalar
    row = np.random.choice(data.shape[0], size = 1)
    col = np.random.choice(data.shape[1], size = 1)

    # inner loop for number of BRK iterations
    for j in np.arange(siter):
        if s == 1:
            # sample index for entry of data matrix
            kaczrow = np.random.choice(lf.shape[0], size = s, p = weightsample(lf, 1), replace = False)
            kaczcol = np.random.choice(rf.shape[1], size = s, p = weightsample(rf, 0), replace = False)
            
            lfactor[row, :] = lfactor[row, :] + (data[row, kaczcol] - lfactor[row, :] @ rfactor[:, kaczcol]) / (np.linalg.norm(rfactor[:, kaczcol])**2) * rfactor[:, kaczcol].T 
            rfactor[:, col] = rfactor[:, col] + (data[kaczrow, col] - lfactor[kaczrow, :] @ rfactor[:, col]) / (np.linalg.norm(lfactor[kaczrow, :])**2) * lfactor[kaczrow, :].T
     
        else:
            # sample s.t. at least one row/column is nonzero
            resample = True
            while(resample):
                kaczrow = np.random.choice(lf.shape[0], size = s, replace = False)
                kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)
                if (np.linalg.norm(lf[kaczrow]) > 0 and np.linalg.norm(rf[:, kaczcol]) > 0):
                    resample = False
                    
            # compute BRK step
            lf[row, :] = lf[row, :] + (data[None, row, kaczcol] - lf[row, :] @ rf[:, kaczcol]) @ np.linalg.pinv(rf[:, kaczcol])
            rf[:, col] = rf[:, col] + np.linalg.pinv(lf[kaczrow, :]) @ (data[kaczrow, col, None] - lf[kaczrow, :] @ rf[:, col])

    return(lf, rf)


def qbrkupdate(data, lf, rf, s, siter):    
    """
    "Quick" Block Randomized Kaczmarz Update 
    
    It approximates the least squares solution at each update. This differs from normal BRK by uniformly sampling 
    of rows/cols rather than performing a weighted sampling, trading approximation quality for computational time.
    """
    # uniform sampling of row and column from data matrix
    # specifying size returns an array rather than a scalar
    row = np.random.choice(data.shape[0], size = 1)
    col = np.random.choice(data.shape[1], size = 1)

    # inner loop for number of BRK iterations
    for j in np.arange(siter):
        # sample index for entry of data matrix
        kaczrow = np.random.choice(lf.shape[0], size = s, replace = False)
        kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)
        
        lf[row, :] = lf[row, :] + np.matmul((data[None, row, kaczcol] - np.matmul(lf[row, :], rf[:, kaczcol])), np.linalg.pinv(rf[:, kaczcol]))
        rf[:, col] = rf[:, col] + np.matmul(np.linalg.pinv(lf[kaczrow, :]), (data[kaczrow, col, None] - np.matmul(lf[kaczrow, :], rf[:, col])))

    return(lf, rf)


def bgsupdate(data, lf, rf, s, siter):
    approx = np.matmul(lf, rf)
    row = np.random.choice(data.shape[0], size = 1)
    col = np.random.choice(data.shape[1], size = 1)

    # inner loop for number of GS iterations
    for j in np.arange(siter):
        if s == 1:
            gsrow = np.random.choice(rf.shape[0], size = s, p = weightsample(rf, 1), replace = False)
            gscol = np.random.choice(lf.shape[1], size = s, p = weightsample(lf, 0), replace = False)
        else:
            resample = True
            while(resample):
                rowsum = 0
                colsum = 0
                gscol = np.random.choice(lf.shape[1], size = s, replace = False)
                gsrow = np.random.choice(rf.shape[0], size = s, replace = False)

                for samplerow in gsrow:
                    rowsum = rowsum + sum(lf[samplerow, :])
                for samplecol in gscol:
                    colsum = colsum + sum(rf[:, samplecol])
                if (rowsum > 0 and colsum > 0):
                    resample = False

        # compute BGS step
        lf[row, :] = lf[row, :] + np.matmul((data[row, :] - np.matmul(lf[row, :], rf)), np.matmul(np.linalg.pinv(rf[gsrow, :]), np.eye(k)[:, gscol].T))
        rf[:, col] = rf[:, col] + np.matmul(np.matmul(np.eye(k)[:, gscol], np.linalg.pinv(lf[:, gscol])), (data[:, col] - np.matmul(lf, rf[:, col])))
    return(lf, rf)
