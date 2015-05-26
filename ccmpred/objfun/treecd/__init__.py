import numpy as np
import Bio.Phylo

import ccmpred.raw
import ccmpred.counts
import ccmpred.objfun
import ccmpred.objfun.cd
import ccmpred.objfun.treecd.cext


class TreeContrastiveDivergence(ccmpred.objfun.cd.ContrastiveDivergence):

    def __init__(self, msa, tree, seq0, id0, weights, lambda_single, lambda_pair, mutation_rate):
        super(TreeContrastiveDivergence, self).__init__(msa, weights, lambda_single, lambda_pair, len(tree.get_terminals()))

        tree = split_tree(tree, id0)

        self.tree = tree
        self.seq0 = seq0

        tree_bfs = [c for c in bfs_iterator(tree.clade)]

        self.n_children = np.array([len(c.clades) for c in tree_bfs], dtype='uint32')
        self.branch_lengths = np.array([c.branch_length for c in tree_bfs], dtype=np.dtype('float64'))

        self.mutation_rate = mutation_rate
        self.n_vertices = len(tree_bfs)

    def init_sample_alignment(self):
        return np.empty_like(self.msa, dtype="uint8")

    @classmethod
    def init_from_raw(cls, msa, weights, raw, tree, seq0, id0, lambda_single=1e4, lambda_pair=lambda msa: (msa.shape[1] - 1) * 0.2, mutation_rate=1):
        res = cls(msa, tree, seq0, id0, weights, lambda_single, lambda_pair, mutation_rate)

        if msa.shape[1] != raw.ncol:
            raise Exception('Mismatching number of columns: MSA {0}, raw {1}'.format(msa.shape[1], raw.ncol))

        x_single = raw.x_single
        x_pair = np.transpose(raw.x_pair, (0, 2, 1, 3))
        x = np.hstack((x_single.reshape((-1,)), x_pair.reshape((-1),)))

        res.centering_x_single[:] = x_single

        return x, res

    def sample_sequences(self, x):
        return ccmpred.objfun.treecd.cext.mutate_along_tree(self.msa_sampled, self.n_children, self.branch_lengths, x, self.n_vertices, self.seq0, self.mutation_rate)


def bfs_iterator(clade):
    """Breadth-first iterator along a tree clade"""

    def inner(clade):
        yield from clade.clades
        for c in clade.clades:
            yield from inner(c)

    yield clade
    yield from inner(clade)


def split_tree(tree, id0):
    """Reroot tree so that the clades in id0 are direct descendants of the root node"""
    id_to_node = dict((cl.name, cl) for cl in bfs_iterator(tree.clade))

    new_tree = Bio.Phylo.BaseTree.Tree()
    new_tree.clade.clades = [id_to_node[i] for i in id0]

    for cl in new_tree.clade.clades:
        cl.branch_length = 0

    new_tree.clade.branch_length = 0

    return new_tree
