import numpy as np

def leapfrog(q, dlogq, p, dlogp, step_size=0.3, num_steps=1, run_until_cycle=False):
    # for storing trajectory
    Ps = []
    Qs = []
    
    # create copy of state
    p = np.array(p)
    q = np.array(q)
    Ps += [p]
    Qs += [q]
    
    # half momentum update
    p = p - (step_size / 2) * -dlogq(q)
    
    # alternate full variable and momentum updates
    i = 0
    while True:
        q = q + step_size * -dlogp(p)
        Qs += [q]

        # precompute since used for two half-steps
        dlogq_eval = dlogq(q)

        #  first half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        # store p as now fully updated
        Ps += [p]
        
        # check whether to stop (epsilon close to start point) and maybe stop
        if i >= num_steps-1:
            if run_until_cycle:
                if np.sqrt(np.linalg.norm(Ps[0] - Ps[-1]) ** 2 + np.linalg.norm(Qs[0] - Qs[-1]) ** 2) < step_size:
                    break
            else:
                break
        
        # second half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        i += 1
    Qs = np.asarray(Qs)
    Ps = np.asarray(Ps)
    return Qs, Ps
