import numpy as np
import Bio.Phylo


def bfs_iterator(clade):
    """Breadth-first iterator along a tree clade"""

    def inner(clade):
        yield from clade.clades
        for c in clade.clades:
            yield from inner(c)

    yield clade
    yield from inner(clade)


def parse_tree(tree_file, tree_format='newick'):
    tree = Bio.Phylo.read(tree_file, tree_format)
    tree_bfs = [c for c in bfs_iterator(tree.clade)]

    n_children = np.array([len(c.clades) for c in tree_bfs], dtype=int)
    branch_lengths = np.array([c.branch_length for c in tree_bfs], dtype=np.dtype('float64'))

    return n_children, branch_lengths


if __name__ == '__main__':

    nc, bl = parse_tree("data/demo.dnd")

    print(nc)
    print(bl)
