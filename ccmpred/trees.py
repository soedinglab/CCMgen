from Bio.Phylo.BaseTree import Tree, Clade


def create_binary_tree(splits, depth, root_name=""):
    depth_per_clade = float(depth) / splits

    def fill_tree_rec(parent, splits):
        if splits == 0:
            return

        c1 = Clade(name=parent.name + "A", branch_length=depth_per_clade)
        c2 = Clade(name=parent.name + "B", branch_length=depth_per_clade)

        fill_tree_rec(c1, splits - 1)
        fill_tree_rec(c2, splits - 1)

        parent.clades = [c1, c2]

    t = Tree(rooted=False)
    t.clade.name = root_name
    t.clade.branch_length = 0
    fill_tree_rec(t.clade, splits)

    return t


def create_star_tree(leaves, depth, root_name=""):

    t = Tree(rooted=False)
    t.clade.name = root_name
    t.clade.branch_length = 0

    t.clade.clades = [
        Clade(name="C{0}".format(i), branch_length=depth)
        for i in range(leaves)
    ]

    return t
