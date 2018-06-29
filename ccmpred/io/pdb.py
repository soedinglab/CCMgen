from Bio.PDB import PDBParser
import numpy as np

def read_pdb(pdbfile):
    '''
    Read a PDB file as structure file with BIO.PDB

    :param pdbfile: path to pdb file
    :return:  structure
    '''

    parser = PDBParser()
    structure = parser.get_structure('pdb', pdbfile)

    return structure

def calc_residue_dist(residue_one, residue_two, distance_definition="Cb"):
    '''
        Calculate euclidian distance between C-beta (C-alpha in case of Glycine/missing C-beta)
        atoms of oth residues
    :param residue_one: BIO.PDB residue object 1
    :param residue_two: BIO.PDB residue object 2
    :return: float euclidian distance between residues
    '''

    if distance_definition == "Cb":

        if residue_one.has_id("CB"):
            residue_one_atom = residue_one["CB"]
        else:
            residue_one_atom = residue_one["CA"]

        if residue_two.has_id("CB"):
            residue_two_atom = residue_two["CB"]
        else:
            residue_two_atom = residue_two["CA"]

        diff_vector = residue_one_atom.coord - residue_two_atom.coord
        diff = np.sqrt(np.sum(diff_vector * diff_vector))
    else:
        diff_list = []
        for atom_1 in [atom for atom in residue_one if atom.name not in ['N', 'O', 'C']]:
            for atom_2 in [atom for atom in residue_two if atom.name not in ['N', 'O', 'C']]:
                diff_vector = atom_1.coord - atom_2.coord
                diff_list.append(np.sqrt(np.sum(diff_vector * diff_vector)))

        diff = np.min(diff_list)

    return diff

def distance_map(pdb_file, L=None, distance_definition="Cb"):
    '''
    Compute the distances between Cbeta (Calpha for Glycine) atoms of all residue pairs

    :param pdb_file: PDB file (first chain of first model will be used)
    :return: LxL numpy array with distances (L= protein length)
    '''

    structure = read_pdb(pdb_file)
    structure.get_list()
    model = structure[0]
    chain = model.get_list()[0]

    # due to missing residues in the pdb file (or additionally solved??)
    # protein length L can differ from len(chain.get_list())
    if L is None:
        L = chain.get_list()[-1].id[1]

    distance_map = np.full((L, L), np.NaN)

    residues = chain.get_list()
    for i in range(np.min([L, len(chain.get_list())])):
        for j in range(np.min([L, len(chain.get_list())])):
            residue_one = residues[i]
            residue_two = residues[j]
            distance_map[residue_one.id[1] - 1, residue_two.id[1] - 1] = calc_residue_dist(residue_one, residue_two, distance_definition)

    return distance_map