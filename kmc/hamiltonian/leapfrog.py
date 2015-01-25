import numpy as np


def leapfrog(q, dlogq, p=None, dlogp=None, step_size=0.3, num_steps=1):
    assert p is None and dlogp is None or p is not None and dlogp is not None
    
    if p is None or dlogp is None:
        p = np.random.randn(len(q))
        dlogp = lambda x:-x
    
    # for storing trajectory
    Ps = np.zeros((num_steps + 1, len(p)))
    Qs = np.zeros(Ps.shape)
    
    # create copy of state
    p = np.array(p)
    q = np.array(q)
    Ps[0] = p
    Qs[0] = q
    
    # half momentum update
    p = p - (step_size / 2) * -dlogp(q)
    
    # alternate full variable and momentum updates
    for i in range(num_steps):
        q = q + step_size * -dlogq(p)
        Qs[i + 1] = q

        # precompute since used for two half-steps
        dlogp_eval = dlogp(q)

        #  second half momentum update
        p = p - (step_size / 2) * -dlogp_eval
        
        # store p as now fully updated
        Ps[i + 1] = p

        # first half momentum update if not last iteration
        if i != num_steps - 1:
            p = p - (step_size / 2) * -dlogp_eval
    
    return Qs, Ps
