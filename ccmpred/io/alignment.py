import numpy as np

import ccmpred.counts


def read_msa(f, format, return_indices=True):
    if format == 'psicov':
        return read_msa_psicov(f, return_indices)
    else:
        return read_msa_biopython(f, format, return_indices)


def read_msa_biopython(f, format, return_indices=True):
    import Bio.AlignIO as aio

    records = list(aio.read(f, format))

    msa = [r.seq.tostring() for r in records]
    msa = np.array([[ord(c) for c in x.strip()] for x in msa], dtype=np.uint8)

    if return_indices:
        ccmpred.counts.index_msa(msa, in_place=True)

    return msa


def read_msa_psicov(f, return_indices=True):

    if isinstance(f, str):
        with open(f, 'r') as o:
            msa = o.readlines()
    else:
        msa = f

    msa = np.array([[ord(c) for c in x.strip()] for x in msa], dtype=np.uint8)

    if return_indices:
        ccmpred.counts.index_msa(msa, in_place=True)

    return msa


def write_msa_psicov(f, msa, is_indices=True):
    if is_indices:
        msa = ccmpred.counts.char_msa(msa)

    f.write("\n".join(["".join(chr(cell) for cell in row) for row in msa]))
