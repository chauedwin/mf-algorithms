#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import multiprocessing as mp


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
            row = np.random.choice(data.shape[0], p = weightsample(approx, 1))
            col = np.random.choice(data.shape[1], p = weightsample(approx, 0))
            
            # inner loop for number of RK iterations
            for j in np.arange(kacziter):
                # sample index for entry of data matrix
                kaczrow = np.random.choice(lfactor.shape[0], p = weightsample(lfactor, 1))
                kaczcol = np.random.choice(rfactor.shape[1], p = weightsample(rfactor, 0))
                
                # compute RK step
                lfactor[row, :] = lfactor[row, :] + (data[row, kaczcol] - np.matmul(lfactor[row, :], rfactor[:, kaczcol])) / (np.linalg.norm(rfactor[:, kaczcol])**2) * rfactor[:, kaczcol] 
                rfactor[:, col] = rfactor[:, col] + (data[kaczrow, col] - np.matmul(lfactor[kaczrow, :], rfactor[:, col])) / (np.linalg.norm(lfactor[kaczrow, :])**2) * lfactor[kaczrow, :]
     
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


# In[1]:


def alstest(data, k, niter, reinit = 1):
    A, S, e = als(data, k = k, niter = niter, reinit = reinit)
    approx = np.matmul(A, S)
    return((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[2]:


def rktest(data, k, niter, kacziter, reinit = 1):
    A, S, error = rk(data, k = 4, niter = niter, kacziter = kacziter, reinit = reinit)
    approx = np.matmul(A, S)
    return((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[ ]:


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


# In[ ]:


def read(filename): 
    with open(filename, 'r') as f:
        l = f.read().split(',')
    return(l)


# In[8]:


def alswrite(data, k, niter, q):
    A, S, e = als(data, k = k, niter = niter)
    approx = np.matmul(A, S)
    q.put((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[ ]:


def rkwrite(data, k, niter, kacziter, q):
    A, S, error = rk(data, k = k, niter = niter, kacziter = kacziter)
    approx = np.matmul(A, S)
    q.put((np.linalg.norm(data - approx) / np.linalg.norm(data)))


# In[ ]:


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


# In[ ]:


def rkmp(data, k, niter, kacziter, cores = mp.cpu_count()): 
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(cores)

    #put listener to work first
    watcher = pool.apply_async(listener, (q, filename))

    #fire off workers
    jobs = []
    for i in range(loop):
        job = pool.apply_async(rkwrite, (data, 4, 100, 1000, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


# In[9]:





# In[ ]:




