import numpy as np
import ccmpred.counts
import Bio.AlignIO as aio

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV-"

def read_msa(f, format, return_indices=True, return_identifiers=False):
    if format == 'psicov':
        return read_msa_psicov(f, return_indices, return_identifiers)
    else:
        return read_msa_biopython(f, format, return_indices, return_identifiers)

def read_msa_biopython(f, format, return_indices=True, return_identifiers=False):

    records = list(aio.read(f, format))

    msa = [str(r.seq) for r in records]
    msa = np.array([[ord(c) for c in x.strip()] for x in msa], dtype=np.uint8)

    if return_indices:
        ccmpred.counts.index_msa(msa, in_place=True)

    if return_identifiers:
        identifiers = [r.name for r in records]
        return msa, identifiers
    else:
        return msa

def read_msa_psicov(f, return_indices=True, return_identifiers=False):

    if isinstance(f, str):
        with open(f, 'r') as o:
            msa = o.readlines()
    else:
        msa = f

    for i, line in enumerate(msa):
        if ">" in line:
            raise Exception("Line number {0} contains a '>' - please set the correct alignment format!:\n{1}".format(i + 1, line))

    msa = np.array([[ord(c) for c in x.strip()] for x in msa], dtype=np.uint8)

    if return_indices:
        ccmpred.counts.index_msa(msa, in_place=True)

    if return_identifiers:
        identifiers = ["seq{0}".format(i) for i in range(msa.shape[0])]
        return msa, identifiers
    else:
        return msa


def write_msa(f, msa, ids, format, is_indices=True, descriptions=None):

    if format == 'psicov':
        write_msa_psicov(f, msa, is_indices=is_indices)
    else:
        write_msa_biopython(f, msa, ids, format, is_indices=is_indices, descriptions=descriptions)

def write_msa_psicov(f, msa, is_indices=True):

    if is_indices:
        msa = ccmpred.counts.char_msa(msa)

    f.write("\n".join(["".join(chr(cell) for cell in row) for row in msa]))

def write_msa_biopython(f, msa, ids, format, is_indices=True, descriptions=None):
    import Bio.SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    if is_indices:
        msa = ccmpred.counts.char_msa(msa)

    if descriptions is None:
        descriptions = ["" for _ in range(msa.shape[0])]

    msa = ["".join(chr(c) for c in row) for row in msa]

    records = [
        SeqRecord(Seq(seq, Bio.Alphabet.IUPAC.protein), id=id, description=desc)
        for seq, id, desc in zip(msa, ids, descriptions)
    ]

    Bio.SeqIO.write(records, f, format)
