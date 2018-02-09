import ccmpred.objfun.cd.cext

def gibbs_sample_sequences(x, msa_sampled, gibbs_steps):
    return ccmpred.objfun.cd.cext.gibbs_sample_sequences(msa_sampled, x, gibbs_steps)