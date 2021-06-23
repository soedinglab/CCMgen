import numpy as np
import ccmpred.raw

def write_new_tree_for_ccmgen(file_coupling : str, file_name : str):
    
    raw = ccmpred.raw.parse_msgpack(file_coupling)
    Field = raw.x_single
    size_prot = Field.shape[0]
    
    sequence = ''.join([random.choice('ACDEFGHIKLMNPQRSTVWY-') for x in range(size_prot)]) 

    print("Random sequence : %s"%sequence)
    
    record = SeqRecord(
        Seq(sequence),
        id="ID_0.1",
        name="RandomSequences",
        description="Random Sequences for the root of a phylogeny tree",
    )

    SeqIO.write(record, file_name, "fasta")

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    write_new_tree_for_ccmgen(*sys.argv[1:])

