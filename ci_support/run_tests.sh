set -e
ccmgen --tree-newick ci_support/phylo.newick --aln-format psicov --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --aln-format fasta --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa

## New test
coupling = ${ci_support/mrf_params.braw.gz}
seq0_file= ${ci_support/seq0_file.fasta}

python3 random_start_sequence.py $coupling $seq0_file

ccmgen --tree-newick ci_support/phylo.newick --seq0-file $seq0_file  --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --seq0-file $seq0_file  --mutation-rate-neff --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
