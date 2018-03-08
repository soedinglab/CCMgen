import Bio.Phylo.BaseTree
import Bio.Phylo
import numpy as np
import ccmpred.sampling


class CCMTree(object):
    """This class represents an empty phylogenetic tree according to some specific topology"""

    def __init__(self):
        """Initialise all class attributes"""

        self.nseq = None
        self.id0 = ["root"]
        self.ids = None
        self.n_children = None
        self.branch_lengths = None
        self.n_vertices = None
        self.n_leaves = None
        self.tree = None

    def specify_tree(self, nseq, tree_file=None, tree_source=None):
        """
        Parameters
        ----------
        nseq: int
            Specifies the number of leave nodes representing sequences
        tree_file: str or None, optional
            path to a newick type tree topology file
        tree_source: str or None, optional
            specifies a tree topology [star|binary]

        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        self.nseq = nseq

        if tree_source == "binary":
            self.tree = create_binary_tree(self.nseq, root_name=self.id0[0])
        elif tree_source == "star":
            self.tree = create_star_tree(self.nseq, root_name=self.id0[0])
        elif tree_file is not None:
            try:
                self.tree = Bio.Phylo.read(tree_file, "newick")
            except ValueError as e:
                print("Error while reading tree file {0} : {1}".format(tree_file, e))
                return False
            except OSError as e:
                print("Error while reading tree file {0} : {1}".format(tree_file, e))
                return False

        if self.tree is not None:
            # prepare tree topology
            tree_split = split_tree(self.tree, self.id0)
            tree_bfs = [c for c in bfs_iterator(tree_split.clade)]

            self.n_children = np.array([len(c.clades) for c in tree_bfs], dtype='uint64')
            self.branch_lengths = np.array([c.branch_length for c in tree_bfs], dtype=np.dtype('float64'))
            self.n_vertices = len(tree_bfs)
            self.n_leaves = len(tree_split.get_terminals())
            self.ids = [l.name for l in tree_split.get_terminals()]

            depth_min, depth_max = get_child_depth_range(tree_split.clade)
            print("Created {0} tree with {1} leaves, depth_min={2:.4e}, depth_max={3:.4e}\n".format(
                tree_source, self.n_leaves, depth_min, depth_max))

        return True


def split_tree(tree, id0):
    """Reroot tree so that the clades in id0 are direct descendants of the root node"""

    id_to_node = dict((cl.name, cl) for cl in bfs_iterator(tree.clade))

    new_tree = Bio.Phylo.BaseTree.Tree()
    new_tree.clade.clades = [id_to_node[i] for i in id0]

    for cl in new_tree.clade.clades:
        cl.branch_length = 0

    new_tree.clade.branch_length = 0

    return new_tree


def bfs_iterator(clade):
        """Breadth-first iterator along a tree clade"""

        def inner(clade):
            for c in clade.clades:
                yield c

            for c in clade.clades:
                for ci in inner(c):
                    yield ci

        yield clade

        for ci in inner(clade):
            yield ci


def get_child_depth_range(clade):
        """Return the minimum and maximum child depth"""
        level = [(0, clade)]

        mn = float('inf')
        mx = float('-inf')
        while level:
            new_level = []

            for d, parent in level:
                dc = d + parent.branch_length

                if parent.clades:
                    for c in parent.clades:
                        new_level.append((dc, c))
                else:
                    mn = min(mn, dc)
                    mx = max(mx, dc)

            level = new_level

        return mn, mx


def get_seq0_mrf(x, ncol, gibbs_steps):
    """
        Specify the root sequence in the tree representing the common ancestor.

        A new sequence of length NCOL will be sampled from a poly-A sequence of length NCOL
        according to a Markov-Random-Field (MRF aka Potts) model specified by parameters X.

        Parameters
        ----------
        x : ndarray
            1D float containing concatenation of single and pair potentials specifiying the MRF
        ncol : int
            protein/sequence length
        gibbs_steps: int
            number of Gibbs steps used in Gibbs sampling procedure
            (one Gibbs step corresponds to sampling a new amino acid for every position)

        Returns
        -------
        ndarray
            1D integer array representing the newly sampled sequence

        """

    # generate a poly-A alignment
    seq0 = np.zeros((1, ncol), dtype="uint8")

    # gibbs sample a new sequence
    seq0 = ccmpred.sampling.gibbs_sample_sequences(x, seq0, gibbs_steps)

    return seq0


def create_binary_tree(nseqs, depth=1, root_name=""):
    """
        Create a binary tree topology.

        The depth of the tree is specified by DEPTH and the number of leave nodes by NSEQS (should be a power of 2).


        Parameters
        ----------
        nseqs : int
            the number of leave nodes that represent sequences
        depth : int, optional(default=1)
            the depth of the tree
        root_name: str, optional(default="")
            name of the root sequence

        Returns
        -------
        Bio.Phylo.BaseTree.Tree
            topology of a binary tree

        """

    splits = np.ceil(np.log2(nseqs))

    depth_per_clade = float(depth) / splits

    def fill_tree_rec(parent, splits):
        if splits == 0:
            return

        c1 = Bio.Phylo.BaseTree.Clade(name=parent.name + "A", branch_length=depth_per_clade)
        c2 = Bio.Phylo.BaseTree.Clade(name=parent.name + "B", branch_length=depth_per_clade)

        fill_tree_rec(c1, splits - 1)
        fill_tree_rec(c2, splits - 1)

        parent.clades = [c1, c2]

    t = Bio.Phylo.BaseTree.Tree(rooted=False)
    t.clade.name = root_name
    t.clade.branch_length = 0
    fill_tree_rec(t.clade, splits)

    return t


def create_star_tree(nseqs, depth=1, root_name=""):
    """
        Create a star tree topology.

        The depth of the tree is specified by DEPTH and the number of leave nodes by NSEQS (should be a power of 2).


        Parameters
        ----------
        nseqs : int
            the number of leave nodes that represent sequences
        depth : int, optional(default=1)
            the depth of the tree
        root_name: str, optional(default="")
            name of the root sequence

        Returns
        -------
        Bio.Phylo.BaseTree.Tree
            topology of a star tree

        """

    t = Bio.Phylo.BaseTree.Tree(rooted=False)
    t.clade.name = root_name
    t.clade.branch_length = 0

    t.clade.clades = [
        Bio.Phylo.BaseTree.Clade(name="C{0}".format(i), branch_length=depth)
        for i in range(nseqs)
    ]

    return t
