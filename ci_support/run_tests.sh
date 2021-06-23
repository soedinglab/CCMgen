set -e
ccmgen --tree-newick ci_support/phylo.newick --aln-format psicov --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --aln-format fasta --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa

## New test
python ci_support/random_start_sequence.py exemple/1atzA.braw.gz ci_support/seq0_file.fasta

ccmgen --tree-newick ci_support/phylo.newick --seq0-file ci_support/seq0_file.fasta  --mutation-rate 1 --num-threads 1 exemple/1atzA.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --seq0-file ci_support/seq0_file.fasta  --mutation-rate-neff --num-threads 1 --alnfile exemple/1atzA.fas  exemple/1atzA.braw.gz sequences.msa

