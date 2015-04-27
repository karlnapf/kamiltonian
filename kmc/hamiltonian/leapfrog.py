import numpy as np

def leapfrog(q, dlogq, p, dlogp, step_size=0.3, num_steps=1):
    # for storing trajectory
    Ps = np.zeros((num_steps + 1, len(p)))
    Qs = np.zeros(Ps.shape)
    
    # create copy of state
    p = np.array(p.copy())
    q = np.array(q.copy())
    Ps[0] = p
    Qs[0] = q
    
    # half momentum update
    p = p - (step_size / 2) * -dlogq(q)
    
    # alternate full variable and momentum updates
    for i in range(num_steps):
        q = q + step_size * -dlogp(p)
        Qs[i + 1] = q

        # precompute since used for two half-steps
        dlogq_eval = dlogq(q)

        #  first half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        # store p as now fully updated
        Ps[i + 1] = p
        
        # second half momentum update
        if i != num_steps - 1:
            p = p - (step_size / 2) * -dlogq_eval

    return Qs, Ps

def leapfrog_no_storing(q, dlogq, p, dlogp, step_size=0.3, num_steps=1):
    # create copy of state
    p = np.array(p.copy())
    q = np.array(q.copy())
    
    # half momentum update
    p = p - (step_size / 2) * -dlogq(q)
    
    # alternate full variable and momentum updates
    for i in range(num_steps):
        q = q + step_size * -dlogp(p)

        # precompute since used for two half-steps
        dlogq_eval = dlogq(q)

        #  first half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        # second half momentum update
        if i != num_steps - 1:
            p = p - (step_size / 2) * -dlogq_eval

    return q, p