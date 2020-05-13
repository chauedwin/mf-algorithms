#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import multiprocessing as mp
import random
import re
import sys
import scipy.sparse


# In[2]:


def softprojc(vec, i, c = -1e-5):
    return(np.where(vec < c, c, vec))
def softproji(vec, i):
    return(np.where(vec < 0, (-1 / np.sqrt(i)), vec))


# In[3]:


# mode 1 samples rows
def weightsample(data, mode):
    prob = np.linalg.norm(data, axis=mode)
    return(prob / sum(prob))


# In[4]:


def als(data, k, niter, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for j in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])

        for i in np.arange(niter):            
            # sample random row or column
            row = np.random.randint(data.shape[0])
            col = np.random.randint(data.shape[1])
            
            # perform linear reg update 
            rfactor[:, col] = np.matmul(np.linalg.pinv(lfactor), data[:, col])
            lfactor[row, :] = np.matmul(data[row, :], np.matmul(rfactor.T, np.linalg.inv(np.matmul(rfactor, rfactor.T))))
            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[niter - 1]):
            finalerror = seqerror[niter - 1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    return(lbest, rbest, lowesterror)


# In[5]:


def rk(data, k, niter, kacziter, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):
            approx = np.matmul(lfactor, rfactor)
            
            # weighted sampling of row and column from data approx matrix
            row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
            col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))
            
            # inner loop for number of RK iterations
            for j in np.arange(kacziter):
                # sample index for entry of data matrix
                kaczrow = np.random.choice(lfactor.shape[0], size = 1, p = weightsample(lfactor, 1))
                kaczcol = np.random.choice(rfactor.shape[1], size = 1, p = weightsample(rfactor, 0))

                # compute RK step
                lfactor[row, :] = lfactor[row, :] + (data[row, kaczcol] - np.matmul(lfactor[row, :], rfactor[:, kaczcol])) / (np.linalg.norm(rfactor[:, kaczcol])**2) * rfactor[:, kaczcol].T 
                rfactor[:, col] = rfactor[:, col] + (data[kaczrow, col] - np.matmul(lfactor[kaczrow, :], rfactor[:, col])) / (np.linalg.norm(lfactor[kaczrow, :])**2) * lfactor[kaczrow, :].T
     
            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[niter - 1]):
            finalerror = seqerror[niter - 1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    return(lbest, rbest, lowesterror)


# In[6]:


def brk(data, k, s, niter, kacziter, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):
            approx = np.matmul(lfactor, rfactor)
            
            # weighted sampling of row and column from data matrix
            row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
            col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))
            #row = np.random.choice(data.shape[0], p = weightsample(approx, 1))
            #col = np.random.choice(data.shape[1], p = weightsample(approx, 0))
            
            # inner loop for number of RK iterations
            for j in np.arange(kacziter):
                # sample indices until at least one nonzero row or col
                
                resample = True
                while(resample):
                    rowsum = 0
                    colsum = 0
                    kaczrow = np.random.choice(lfactor.shape[0], size = s, replace = False)
                    kaczcol = np.random.choice(rfactor.shape[1], size = s, replace = False)

                    for samplerow in kaczrow:
                        rowsum = rowsum + sum(lfactor[samplerow, :])
                    for samplecol in kaczcol:
                        colsum = colsum + sum(rfactor[:, samplerow])
                    if (rowsum > 0 and colsum > 0):
                        resample = False

                # compute BRK step
                #kaczrow = np.random.choice(lfactor.shape[0], size = s, replace = False)
                #kaczcol = np.random.choice(rfactor.shape[1], size = s, replace = False)

                lfactor[row, :] = lfactor[row, :] + np.matmul((data[row, kaczcol] - np.matmul(lfactor[row, :], rfactor[:, kaczcol])), np.linalg.pinv(rfactor[:, kaczcol]))
                rfactor[:, col] = rfactor[:, col] + np.matmul(np.linalg.pinv(lfactor[kaczrow, :]), (data[kaczrow, col, None] - np.matmul(lfactor[kaczrow, :], rfactor[:, col])))

            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[niter - 1]):
            finalerror = seqerror[niter - 1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    return(lbest, rbest, lowesterror)


# In[7]:


def gs(data, k, niter, gsiter, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):
            approx = np.matmul(lfactor, rfactor)
            
            # weighted sampling of row and column from data matrix
            #row = np.random.choice(data.shape[0], p = weightsample(approx, 1))
            #col = np.random.choice(data.shape[1], p = weightsample(approx, 0))
            row = np.random.choice(data.shape[0])
            col = np.random.choice(data.shape[1])
            
            # inner loop for number of RK iterations
            for j in np.arange(gsiter):
                # sample indices for entry of data matrix, dont want norms in rk step to be 0
                #gsrow = np.random.choice(rfactor.shape[0], replace = False, p = weightsample(rfactor, 1))
                #gscol = np.random.choice(lfactor.shape[1], replace = False, p = weightsample(lfactor, 0))
                gsrow = np.random.choice(rfactor.shape[0])
                gscol = np.random.choice(lfactor.shape[1])
            
                # compute GS step
                rfactor[:, col] = rfactor[:, col] + np.matmul(lfactor[:, gscol].T, (data[:, col] - np.matmul(lfactor, rfactor[:, col])), np.eye(k)[:, gscol])
                lfactor[row, :] = lfactor[row, :] + np.matmul((data[row, :] - np.matmul(lfactor[row, :], rfactor)), np.matmul(np.array([rfactor[gsrow, :]]).T, np.array([np.eye(k)[gsrow, :]])))
                
            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[niter - 1]):
            finalerror = seqerror[niter - 1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    return(lbest, rbest, lowesterror)


# In[63]:


def bgs(data, k, s, niter, gsiter, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):
            approx = np.matmul(lfactor, rfactor)
            
            # weighted sampling of row and column from data matrix
            #row = np.random.choice(data.shape[0], p = weightsample(approx, 1))
            #col = np.random.choice(data.shape[1], p = weightsample(approx, 0))
            row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
            col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))
            
            
            # inner loop for number of RK iterations
            for j in np.arange(gsiter):
                # sample indices for entry of data matrix, dont want norms in rk step to be 0
                #gsrow = np.random.choice(rfactor.shape[0], size = s, replace = False, p = weightsample(rfactor, 1))
                #gscol = np.random.choice(lfactor.shape[1], size = s, replace = False, p = weightsample(lfactor, 0))
                
                if s == 1:
                    gsrow = np.random.choice(rfactor.shape[0], size = s, p = weightsample(rfactor, 1))
                    gscol = np.random.choice(lfactor.shape[1], size = s, p = weightsample(lfactor, 0))

                else:
                    resample = True
                    while(resample):
                        rowsum = 0
                        colsum = 0
                        gscol = np.random.choice(lfactor.shape[1], size = s, replace = False)
                        gsrow = np.random.choice(rfactor.shape[0], size = s, replace = False)

                        for samplerow in gsrow:
                            rowsum = rowsum + sum(lfactor[samplerow, :])
                        for samplecol in gscol:
                            colsum = colsum + sum(rfactor[:, samplerow])
                        if (rowsum > 0 and colsum > 0):
                            resample = False

                # compute BGS step
                lfactor[row, :] = lfactor[row, :] + np.matmul((data[row, :] - np.matmul(lfactor[row, :], rfactor)), np.matmul(np.linalg.pinv(rfactor[gsrow, :]), np.eye(k)[:, gscol].T))
                rfactor[:, col] = rfactor[:, col] + np.matmul(np.matmul(np.eye(k)[:, gscol], np.linalg.pinv(lfactor[:, gscol])), (data[:, col] - np.matmul(lfactor, rfactor[:, col])))
        
            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[niter - 1]):
            finalerror = seqerror[niter - 1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    return(lbest, rbest, lowesterror)


# In[10]:


def alstest(data, k, niter, reinit = 1):
    A, S, e = als(data, k = k, niter = niter, reinit = reinit)
    approx = np.matmul(A, S)
    return((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[11]:


def rktest(data, k, niter, kacziter, reinit = 1):
    A, S, error = rk(data, k = k, niter = niter, kacziter = kacziter, reinit = reinit)
    approx = np.matmul(A, S)
    return((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[12]:


def brktest(data, k, s, niter, kacziter, reinit = 1):
    A, S, error = brk(data, k = k, s = s,  niter = niter, kacziter = kacziter, reinit = reinit)
    approx = np.matmul(A, S)
    return((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[13]:


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


# In[14]:


def read(filename): 
    with open(filename, 'r') as f:
        l = f.read().split(',')
    return(l)


# In[15]:


def alswrite(data, k, niter, q):
    A, S, e = als(data, k = k, niter = niter)
    approx = np.matmul(A, S)
    q.put((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[16]:


def rkwrite(data, k, niter, kacziter, q):
    A, S, error = rk(data, k = k, niter = niter, kacziter = kacziter)
    approx = np.matmul(A, S)
    q.put((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[17]:


def brkwrite(data, k, s, niter, kacziter, q):
    A, S, error = brk(data, k = k, s = s, niter = niter, kacziter = kacziter)
    approx = np.matmul(A, S)
    q.put((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[18]:


def alsmp(data, k, niter, filename, loop, cores = mp.cpu_count()):
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(alswrite, (data, 4, 100, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# In[19]:


def rkmp(data, k, niter, kacziter, filename, loop, cores = mp.cpu_count()): 
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(rkwrite, (data, k, niter, kacziter, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# In[20]:


def brkmp(data, k, s, niter, kacziter, filename, loop, cores = mp.cpu_count()): 
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(brkwrite, (data, k, s, niter, kacziter, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# In[21]:


def extracterr(tag, errfiles): 
    r = re.compile(".*(" + tag + ").*")
    files = list(filter(r.match, errfiles))
    title = list()
    meanerr = list()
    stderr = list()
    for f in reversed(files):
        title.append("".join(re.findall('[0-9]+k*', f)[0]))
        meanerr.append(np.mean(np.asarray(read(f)[:-1]).astype(float)))
        stderr.append(np.std(np.asarray(read(f)[:-1]).astype(float)))
    return(title, meanerr, stderr)


# In[22]:


def alsupdate(data, lf, rf, s, siter):
    row = np.random.randint(data.shape[0], size = 1)
    col = np.random.randint(data.shape[1], size = 1)
            
    # perform linear reg update 
    for i in np.arange(siter):
        rf[:, col] = np.matmul(np.linalg.pinv(lf), data[:, col])
        lf[row, :] = np.matmul(data[row, :], np.matmul(rf.T, np.linalg.inv(np.matmul(rf, rf.T))))
    return(lf, rf)


# In[23]:


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
            kaczrow = np.random.choice(lf.shape[0], size = s, p = weightsample(lf, 1))
            kaczcol = np.random.choice(rf.shape[1], size = s, p = weightsample(rf, 0))
        else:
            resample = True
            while(resample):
                rowsum = 0
                colsum = 0
                kaczrow = np.random.choice(lf.shape[0], size = s, replace = False)
                kaczcol = np.random.choice(rf.shape[1], size = s, replace = False)

                for samplerow in kaczrow:
                    rowsum = rowsum + sum(lf[samplerow, :])
                for samplecol in kaczcol:
                    colsum = colsum + sum(rf[:, samplerow])
                if (rowsum > 0 and colsum > 0):
                    resample = False

            # compute BRK step
    
        lf[row, :] = lf[row, :] + np.matmul((data[None, row, kaczcol] - np.matmul(lf[row, :], rf[:, kaczcol])), np.linalg.pinv(rf[:, kaczcol]))
        rf[:, col] = rf[:, col] + np.matmul(np.linalg.pinv(lf[kaczrow, :]), (data[kaczrow, col, None] - np.matmul(lf[kaczrow, :], rf[:, col])))

    return(lf, rf)


# In[70]:


def bgsupdate(data, lf, rf, s, siter):
    approx = np.matmul(lf, rf)
    k = lf.shape[1]
            
    # weighted sampling of row and column from data matrix
    # specifying size returns an array rather than a scalar
    row = np.random.choice(data.shape[0], size = 1, p = weightsample(approx, 1))
    col = np.random.choice(data.shape[1], size = 1, p = weightsample(approx, 0))

    # inner loop for number of GS iterations
    for j in np.arange(siter):
        if s == 1:
            gsrow = np.random.choice(rf.shape[0], size = s, p = weightsample(rf, 1))
            gscol = np.random.choice(lf.shape[1], size = s, p = weightsample(lf, 0))
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
                    colsum = colsum + sum(rf[:, samplerow])
                if (rowsum > 0 and colsum > 0):
                    resample = False

        # compute BGS step
        lf[row, :] = lf[row, :] + np.matmul((data[row, :] - np.matmul(lf[row, :], rf)), np.matmul(np.linalg.pinv(rf[gsrow, :]), np.eye(k)[:, gscol].T))
        rf[:, col] = rf[:, col] + np.matmul(np.matmul(np.eye(k)[:, gscol], np.linalg.pinv(lf[:, gscol])), (data[:, col] - np.matmul(lf, rf[:, col])))
    return(lf, rf)


# In[74]:


def mf(data, k, s = 1, niter = 100, siter = 1, solver = 'als', errseq = False, reinit = 1):
    # set to negative one so we can guarantee an update for the first init
    finalerror = -1
    rows = np.empty(niter)
    
    # need to compare final error to overall best and store the overall best
    seqerror = np.empty(niter)
    lowesterror = np.empty(1)
    
    # store overall best factor matrices
    lbest = np.random.rand(data.shape[0], k)
    rbest = np.random.rand(k, data.shape[1])
    
    # for bgs, siter > 1 throws an error 
    if solver == "als":
        f = alsupdate
    if solver == "brk":
        f = brkupdate
    if solver == "bgs":
        f = bgsupdate
    
    for l in np.arange(reinit):
        # randomly initialize the factor matrices
        lfactor = np.random.rand(data.shape[0], k)
        rfactor = np.random.rand(k, data.shape[1])
        
        # outer loop for number of iterations 
        for i in np.arange(niter):          
            lfactor, rfactor = f(data, lfactor, rfactor, s, siter)
            
            # calculate error after update
            seqerror[i] = np.linalg.norm(data - np.matmul(lfactor, rfactor)) / np.linalg.norm(data)
        # update after first init
        if (finalerror == -1):
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
        # if not first, only update if final error is lower than overall best
        elif (finalerror > seqerror[-1]):
            finalerror = seqerror[-1]
            lowesterror = seqerror
            lbest = lfactor
            rbest = rfactor
    if (errseq):
        return(lbest, rbest, lowesterror)
    else:
        return(lbest, rbest, lowesterror[-1])


# In[76]:


def mfwrite(data, k, s, niter, siter, solver, q, errseq = False, reinit = 1):
    A, S, error = mf(data, k, s, niter, siter, solver, errseq, reinit)
    q.put(error)


# In[80]:


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

