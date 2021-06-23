set -e
ccmgen --tree-newick ci_support/phylo.newick --aln-format psicov --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --aln-format fasta --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa

## New test

ccmgen --tree-newick ci_support/phylo.newick --seq0-file ci_support/seq0_file.fasta  --mutation-rate 1 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa
ccmgen --tree-newick ci_support/phylo.newick --seq0-file ci_support/seq0_file.fasta  --mutation-rate-neff 10 --num-threads 1 ci_support/mrf_params.braw.gz sequences.msa

