#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import multiprocessing as mp
import random
import re
import glob
import sys


# In[2]:


# mode 1 samples rows
def weightsample(data, mode):
    prob = np.linalg.norm(data, axis=mode)
    return(prob / sum(prob))


# In[3]:


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


# In[4]:


def read(filename): 
    with open(filename, 'r') as f:
        l = f.read().split(',')
    return(l)


# In[5]:


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


# In[6]:


def alsupdate(data, lf, rf, s, siter):
    row = np.random.randint(data.shape[0], size = 1)
    col = np.random.randint(data.shape[1], size = 1)
            
    # perform linear reg update 
    for i in np.arange(siter):
        rf[:, col] = np.matmul(np.linalg.pinv(lf), data[:, col])
        lf[row, :] = np.matmul(data[row, :], np.matmul(rf.T, np.linalg.inv(np.matmul(rf, rf.T))))
    return(lf, rf)


# In[7]:


def brkupdate(data, lf, rf, s, siter):
    approx = np.matmul(lf, rf)
            
    # weighted sampling of row and column from data matrix
    # specifying size returns an array rather than a scalar
    row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
    col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))

    # inner loop for number of BRK iterations
    for j in np.arange(siter):
        if s == 1:
            # sample index for entry of data matrix
            kaczrow = np.random.choice(lf.shape[0], size = s, p = weightsample(lf, 1), replace = False)
            kaczcol = np.random.choice(rf.shape[1], size = s, p = weightsample(rf, 0), replace = False)
        else:
            # sample st at least one row/column is nonzero
            resample = True
            while(resample):
                rowsum = 0
                colsum = 0
                kaczrow = np.random.choice(lf.shape[0], size = s, replace = False)
                kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)

                for samplerow in kaczrow:
                    rowsum = rowsum + sum(lf[samplerow, :])
                for samplecol in kaczcol:
                    colsum = colsum + sum(rf[:, samplecol])
                if (rowsum > 0 and colsum > 0):
                    resample = False

            # compute BRK step
    
        lf[row, :] = lf[row, :] + np.matmul((data[None, row, kaczcol] - np.matmul(lf[row, :], rf[:, kaczcol])), np.linalg.pinv(rf[:, kaczcol]))
        rf[:, col] = rf[:, col] + np.matmul(np.linalg.pinv(lf[kaczrow, :]), (data[kaczrow, col, None] - np.matmul(lf[kaczrow, :], rf[:, col])))

    return(lf, rf)


# In[8]:


def quickbrkupdate(data, lf, rf, s, siter):    
    # weighted sampling of row and column from data matrix
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


# In[9]:


def bgsupdate(data, lf, rf, s, siter):
    approx = np.matmul(lf, rf)
    k = lf.shape[1]
            
    # weighted sampling of row and column from data matrix
    # specifying size returns an array rather than a scalar
    #print(lf)
    #print(rf)
    #row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
    #col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))
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
        #print(np.matmul((data[row, :] - np.matmul(lf[row, :], rf)), np.matmul(np.linalg.pinv(rf[gsrow, :]), np.eye(k)[:, gscol].T)))
        #print(np.matmul(np.matmul(np.eye(k)[:, gscol], np.linalg.pinv(lf[:, gscol])), (data[:, col] - np.matmul(lf, rf[:, col]))))
        lf[row, :] = lf[row, :] + np.matmul((data[row, :] - np.matmul(lf[row, :], rf)), np.matmul(np.linalg.pinv(rf[gsrow, :]), np.eye(k)[:, gscol].T))
        rf[:, col] = rf[:, col] + np.matmul(np.matmul(np.eye(k)[:, gscol], np.linalg.pinv(lf[:, gscol])), (data[:, col] - np.matmul(lf, rf[:, col])))
        #print(np.linalg.norm(np.eye(data.shape[0]) - np.matmul(np.linalg.pinv(np.matmul(lf, rf)), np.matmul(lf, rf))))

    return(lf, rf)


# In[10]:


# kill when error is 10 times larger than initial
def mf(data, k, s = 1, niter = 100, siter = 1, solver = 'als', errseq = False, reinit = 1):
    
    # assign solver function based on input
    if solver == "als":
        f = alsupdate
    if solver == "brk":
        f = brkupdate
    if solver == "bgs":
        f = bgsupdate
    if solver == "quickbrk":
        f = quickbrkupdate
    
    # make sure s is valid

    if solver == "bgs":
        assert s <= k, "s should be less than k"
    if solver == "quickbrk" or solver == "brk":
        assert s <= min(data.shape[0], data.shape[1]), "s should be less than the dimension"
    
    # set to negative 1 so we can guarantee an update for the first init
    finalerr = -1
    
    # need to compare final error to overall best and store the overall best
    if (errseq):
        seqerr = np.empty(niter)
    else:
        seqerr = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):   
            '''
            # account for inf
            try:
                lfactor, rfactor = f(data, lfactor, rfactor, s, siter)
                # calculate error after update if sequence is requested
                if (errseq):
                    seqerr[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
            
            except:
                if (errseq):
                    seqerr[i] = float("NaN")
                break
            '''
            lfactor, rfactor = f(data, lfactor, rfactor, s, siter)
            # calculate error after update if sequence is requested
            if (errseq):
                seqerr[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)

        # calculate ending error if no sequence needed
        if (errseq == False):
            try:
                seqerr[0] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
            except:
                seqerr[0] = float("NaN")
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
    if (errseq):
        return(lbest, rbest, finalerr)
    else:
        return(lbest, rbest, finalerr[-1])


# In[11]:


def mfwrite(data, k, s, niter, siter, solver, q, errseq = False, reinit = 1):
    A, S, error = mf(data, k, s, niter, siter, solver, errseq, reinit)
    q.put(error)


# In[12]:


def mpmf(data, k, s, niter, siter, solver, filename, loop, cores = mp.cpu_count()):
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(mfwrite, (data, k, s, niter, siter, solver, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# In[13]:


def createmat(dim, k, s):
    np.random.seed(s)
    factor = np.random.choice(4, size=(dim,k), p=np.array([0.7, 0.1, 0.1, 0.1]))
    weight = np.random.randint(0, 2, size=(k, dim))
    data = np.matmul(factor, weight)
    return(data, factor, weight)

