from ccmpred.sampling.neff import evoldist_for_neff, sample_neff, fit_neff_model


def get_child_depth_range(clade):
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
