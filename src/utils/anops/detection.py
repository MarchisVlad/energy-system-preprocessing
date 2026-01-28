import gams.transfer as gt
import numpy as np
import networkx as nx

from anops.defs import INPUTBLOCKS
from anops.score import WhiteScore, PipsScore


def domains_list(ve, m, withalias=True):
    doms = []
    try:
        for d in m[ve].domain:
            if isinstance(d, gt.UniverseAlias):
                dname = "*"
            elif not withalias and isinstance(d, gt.Alias):
                dname = d.alias_with.name
            else:
                dname = d.name
            doms.append(dname)
    except:
        pass

    return doms


def get_domain_freq(velist):
    doms = {}
    for _, v in velist.items():
        for d in v:
            cnt = doms.get(d, 0)
            doms[d] = cnt + 1
    return doms


def get_domain_tuple(ds, velist, m, pf):
    cnt = 0
    rl = []
    for k, v in velist.items():
        if ds <= set(v):
            rl.append(k)
            cnt = cnt + m[k + pf].number_records
    return cnt, rl


def get_element_domain_dict(model_dict, symbols, suffix):
    return {
        k[:-3]: domains_list(k[:-3], symbols)
        for k, _ in model_dict.data.items()
        if k.endswith(suffix)
    }


def generate_partition_sets(
    domain_freq, domain_card, num_equations, num_sets=5
):
    """
    generates the sets of symbols that are used for partitioning the model into
    blocks. These sets are determined based on the frequency in the model and
    the cardinality of the symbol sets. In addition, only single symbols or
    pairs of symbols are used as a partition set.

    Parameters
    ----------
    domain_freq : dict
        a dictionary mapping the symbol to the number of equations where it is
        a controlled domain
    domain_cand : dict
        a dictionary mapping the symbol to the cardinality of the set.
    num_equations : int
        the number of equations in the model
    num_sets : int
        the number of partition sets to find. If there are less than num_sets
        partition sets that can be found, then only the ones found are
        returned.

    Returns
    -------
    list
        a list of lists, where each list is a partition set
    """
    assert len(domain_freq) == len(domain_card)

    freq_ratio = 0.6
    partitions = []
    # creating the partition sets. The symbols must control at least
    # min_freq of the domains and the cardinality of the symbol set is at
    # least min_card. min_freq and min_card are dynamically updated so that
    # at least one partition set is returned.
    while freq_ratio >= 0.0 and len(partitions) < num_sets:
        min_freq = num_equations * freq_ratio
        min_card = 5

        # creating single symbol partition list.
        partitions = [
            [sym]
            for sym, freq in domain_freq.items()
            if freq >= min_freq and domain_card[sym] >= min_card
        ]

        # generating the pairs of symbols for the partitioning set.
        pairsets = [
            [sym1, sym2]
            for i, sym1 in enumerate(domain_freq.keys())
            for j, sym2 in enumerate(domain_freq.keys())
            if domain_freq[sym1] >= min_freq
            and domain_freq[sym2] >= min_freq
            and domain_card[sym1] * domain_card[sym2] >= min_card
            and sym1 != sym2
            and j > i
        ]

        # adding the pair sets to the partitions
        partitions.extend(pairsets)

        # if the number of desired partition sets are not found, then the
        # frequency ratio is decreased.
        if len(partitions) < num_sets:
            freq_ratio -= 0.1

    return partitions[: min(num_sets, len(partitions))]


def build_partition_block_list(all_symbols, partition):
    """
    builds a list of block labels that will identify variables matching the
    partition

    Parameters
    ----------
    all_symbols : pandas container
        all symbols of the model
    partition : list
        the symbols for the partition. This is either singular or a pair

    Returns
    -------
    list
        a list of block lables for identifying the variable blocks
    """
    symbols1 = all_symbols[partition[0]].records.iloc[:, 0].tolist()

    block_list = []
    if len(partition) == 1:
        block_list = [[sym] for sym in symbols1]
    else:
        assert len(partition) == 2
        symbols2 = list(all_symbols[partition[1]].records.iloc[:, 0].tolist())

        block_list = [[sym1, sym2] for sym1 in symbols1 for sym2 in symbols2]

    return block_list


def get_variable_blocks(
    model_dump, model_dict, variables, partitionlist, block_list
):
    """
    creates a list of length of (number of variables) that maps the variables
    to block ids

    Parameters
    ----------
    model_dump : container
        the gdx container representing the model
    variables : dict
        a dictionary mapping variables to their domain
    partitionlist : list
        a list of symbols used for the partitioning
    block_list : list
        a list of labels that are used to identify variable annotations

    Returns
    -------
    int
      the number of blocks actually found
    numpy array
        an array mapping the variable to block
    """
    # initialising an empty array for the j bloc
    var_blocks = np.full(model_dump["j"].records["element_text"].shape[0], 1)

    # looping over all partitions to assign variables to blocks
    partitionset = set(partitionlist)
    for v, doms in variables.items():
        if partitionset <= set(doms):
            for i, regex in enumerate(block_list):
                query = " and ".join(
                    [
                        f'uni_{doms.index(d) + 1} == "{regex[j]}"'
                        for j, d in enumerate(partitionlist)
                        if d in doms
                    ]
                )
                partvars = model_dict[f"{v}_VM"].records.query(query)["uni_0"]
                varidx = (
                    model_dump["j"]
                    .records["uni"][
                        model_dump["j"].records["uni"].isin(partvars)
                    ]
                    .index.tolist()
                )

                var_blocks[varidx] = i + 2

    unique_labels = np.unique(var_blocks)
    for i, v in enumerate(unique_labels):
        # if the index is the same as the label, then no change is needed
        if i + 1 == v:
            continue

        var_blocks[var_blocks == v] = i + 1

    return len(unique_labels) - 1, var_blocks


def get_equation_blocks(
    model_dump, model_dict, equations, partitionlist, block_list
):
    """
    creates a list of length of (number of equations) that maps the equations
    to block ids

    Parameters
    ----------
    model_dump : container
        the gdx container representing the model
    equations : dict
        a dictionary mapping equations to their domain
    partitionlist : list
        a list of symbols used for the partitioning
    block_list : list
        a list of labels that are used to identify variable annotations

    Returns
    -------
    int
      the number of blocks actually found
    numpy array
        an array mapping the variable to block
    """
    # initialising an empty array for the i bloc
    equ_blocks = np.full(
        model_dump["i"].records["element_text"].shape[0],
        len(partitionlist) * 10,
    )

    # looping over all partitions to assign equations to blocks
    partitionset = set(partitionlist)
    for e, doms in equations.items():
        if partitionset <= set(doms):
            for i, regex in enumerate(block_list):
                query = " and ".join(
                    [
                        f'uni_{doms.index(d) + 1} == "{regex[j]}"'
                        for j, d in enumerate(partitionlist)
                        if d in doms
                    ]
                )
                partequs = model_dict[f"{e}_EM"].records.query(query)["uni_0"]
                equidx = (
                    model_dump["i"]
                    .records["uni"][
                        model_dump["i"].records["uni"].isin(partequs)
                    ]
                    .index.tolist()
                )

                equ_blocks[equidx] = i + 2

    # reindexing the equations after identifying the actual number of blocks
    unique_labels = np.unique(equ_blocks)
    for i, e in enumerate(unique_labels):
        blockid = i + 2
        # if the index is the same as the label, then no change is needed
        if blockid == e:
            continue

        equ_blocks[equ_blocks == e] = blockid

    num_blocks = len(unique_labels)
    equ_blocks[equ_blocks == num_blocks + 1] = num_blocks + 2

    return len(unique_labels), equ_blocks


def vblock_to_eblock(equ_name, var_blocks, blocks, mergecand):
    arr = np.unique(var_blocks)
    num_link = np.count_nonzero(arr == 1)

    # Check if this the equation purely linking
    if np.all(arr == 1):
        return blocks + 2

    # Remove all linking variables
    arr = np.setdiff1d(arr, [1])

    # If length is one, the equation is assigned to a block, otherwise to the last stage
    if len(arr) == 1:
        return arr[0]
    elif len(arr) == 2:
        if tuple(arr) in mergecand:
            mergecand[tuple(arr)] += num_link
        else:
            mergecand[tuple(arr)] = num_link

    return blocks + 2


def eblock_to_vblock(var_name, equ_blocks, blocks, mergecand):
    arr = np.unique(equ_blocks)

    # Check if this the variable is master only
    if np.all(arr == blocks + 2):
        return 1

    # Remove all linking equations
    arr = np.setdiff1d(arr, [blocks + 2])

    # If length is one, the equation is assigned to a block, otherwise to the last stage
    if len(arr) == 1:
        return arr[0]
    else:
        return 1


def get_corresponding_blocks(
    model_dump, block_labels, blocks, input_block_type
):
    """
    computes the block labels for the corresponding problem structure. If a
    variable labelling is input, then the equation block labels are computed.
    Alternatively, if the equation labelling is input, then the variable block
    labels are computed

    Parameters
    ----------
    model_dump : container
        the gdx container representing the model
    block_labels : numpy array
        an array of block label for the input type
    blocks : int
        the number of blocks
    input_block_type : enum
        indicates whether the input blocks are for the variables or equations

    Returns
    -------
    numpy array
        an array that given the variable or equation block labels.
    """
    source_type = "i" if input_block_type == INPUTBLOCKS.EQUATIONS else "j"
    target_type = "j" if input_block_type == INPUTBLOCKS.EQUATIONS else "i"

    elem_to_block = {
        i: j
        for i, j in zip(model_dump[source_type].records["uni"], block_labels)
    }

    # Get A matrix
    source_grouping = (
        model_dump["A"]
        .records.groupby(target_type, observed=True)[source_type]
        .apply(list)
        .values
    )

    target_source_blocks = [
        [elem_to_block.get(e) for e in elems] for elems in source_grouping
    ]

    mapping_func = (
        vblock_to_eblock
        if input_block_type == INPUTBLOCKS.VARIABLES
        else eblock_to_vblock
    )

    # Get equation annotation
    merge_cands = {}
    target_labels = np.array(
        [
            mapping_func(i, source_blocks, blocks, merge_cands)
            for i, source_blocks in enumerate(target_source_blocks)
        ]
    ).astype(float)

    return target_labels, merge_cands


def variable_blocks_from_equations(model_dump, equ_labels, blocks):
    return get_corresponding_blocks(
        model_dump, equ_labels, blocks, INPUTBLOCKS.EQUATIONS
    )


def equation_blocks_from_variables(model_dump, var_labels, blocks):
    return get_corresponding_blocks(
        model_dump, var_labels, blocks, INPUTBLOCKS.VARIABLES
    )


def merge_blocks(block_labels, merge_cands):
    """
    finds the minimum overlapping set of merge candidates, also known as a
    matching in the nodes. The matching set is then used to merge the blocks.
    The block labels will be updated, and resorted.

    Parameters
    ----------
    block_labels : numpy array
        labels assigning variable or equations to blocks
    merge_cands : list of lists
        a list of pairs that are potential merge candidates for blocks

    Returns
    -------
    numpy array
        a new labelling for the blocks
    """
    match_graph = nx.Graph(map(tuple, merge_cands))
    merge_pairs = sorted(nx.maximal_matching(match_graph))

    for p in merge_pairs:
        block_labels[block_labels == p[1]] = p[0]

    # TODO: need to check the difference between variable and equation labels
    unique_labels = np.unique(block_labels)
    for i, l in enumerate(unique_labels):
        # if the index is the same as the label, then no change is needed
        if i + 1 == l:
            continue

        block_labels[block_labels == l] = i + 1

    return len(unique_labels) - 1, block_labels


def merge_adjacent_blocks(block_labels, aggregation_rate):
    """
    merges pairs of blocks that are adjacent. The motivation is that if there
    are a large number of blocks, it is possible that two adjacent blocks are
    related.

    Parameters
    ----------
    block_labels : numpy array
        labels assigning variable or equations to blocks
    aggregation_rate : int
        the number of blocks to aggregate together

    Returns
    -------
    numpy array
        a new labelling for the blocks
    """
    unique_labels = np.unique(block_labels)

    num_labels = len(unique_labels) - 2
    new_label = 2
    for i, l in enumerate(unique_labels[1:-1]):
        if (
            i + 1
        ) % aggregation_rate == 0 and i + aggregation_rate < num_labels:
            new_label += 1

        block_labels[block_labels == l] = new_label

    # relabelling the linking constraint label
    unique_labels = np.unique(block_labels)
    block_labels[block_labels == unique_labels[-1]] = len(unique_labels)

    return len(unique_labels) - 1, block_labels


def move_linking_to_block(
    model_dump, target_blocks, source_blocks, num_blocks, source_block_type
):
    """
    inspects the linking variables and identifies whether they can be moved
    into a block

    Parameters
    ----------
    model_dump : container
        the gdx container representing the model
    target_blocks : numpy array
        an array of block label for the target type
    source_blocks : numpy array
        an array of block labels for the source type
    num_blocks : int
        the number of blocks identified
    source_block_type : enum
        indicates whether the input blocks are for the variables or equations

    Returns
    -------
    int
        the number of linking variables that have been moved to a block
    numpy array
        an array of the new variable labelling
    numpy array
        an array of the new equation labelling
    """
    source_type = "i" if source_block_type == INPUTBLOCKS.EQUATIONS else "j"
    target_type = "j" if source_block_type == INPUTBLOCKS.EQUATIONS else "i"

    linking_label = (
        1 if source_block_type == INPUTBLOCKS.EQUATIONS else num_blocks + 2
    )
    alt_linking_label = 1 if linking_label == num_blocks + 2 else num_blocks + 2

    elem_to_block = {
        i: j
        for i, j in zip(model_dump[source_type].records["uni"], source_blocks)
    }

    # Get A matrix
    source_grouping = (
        model_dump["A"]
        .records.groupby(target_type, observed=True)[source_type]
        .apply(list)
        .values
    )

    target_source_blocks = [
        [elem_to_block.get(e) for e in elems]
        if target_blocks[i] == linking_label
        else None
        for i, elems in enumerate(source_grouping)
    ]

    assert len(target_blocks) == len(source_grouping)

    def relabel_linking(source_blocks, diff):
        unique_blocks = np.unique(source_blocks)
        if len(unique_blocks) > 2:
            return linking_label

        unique_blocks = np.setdiff1d(unique_blocks, alt_linking_label)
        if len(unique_blocks) == 1:
            diff[0] += 1
            return unique_blocks[0]

        return linking_label

    # Get equation annotation
    diff = [0]
    target_labels = np.array(
        [
            relabel_linking(source_blocks, diff)
            if source_blocks is not None
            else target_blocks[i]
            for i, source_blocks in enumerate(target_source_blocks)
        ]
    ).astype(float)

    target_name = (
        "variables"
        if source_block_type == INPUTBLOCKS.EQUATIONS
        else "equations"
    )
    print(f"{diff[0]} linking {target_name} moved to blocks")
    return diff[0], target_labels


def frequency_detection(
    model_dump, model_dict, all_symbols, num_partition_sets=5, only_best=True
):
    """
    detects structure in the model instance by inspecting the frequency of the
    symbols for the equations. In addition, the symbols with a high frequency
    must also have many elements in their set.

    Parameters
    ----------

    Returns
    -------
    """
    # getting the set of equations in the model
    equations = get_element_domain_dict(model_dict, all_symbols, "_EM")

    # getting the set of variables in the model
    variables = get_element_domain_dict(model_dict, all_symbols, "_VM")

    # extracting the domains and computing their frequency in the model
    domain_freq = {
        k: e
        for k, e in sorted(
            get_domain_freq(equations).items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    # extracting the cardinality of the domains
    domain_card = {
        k: all_symbols[k].number_records
        for k, e in get_domain_freq(equations).items()
    }

    partitions = generate_partition_sets(
        domain_freq, domain_card, len(equations), num_partition_sets
    )

    # generating the variable annotations for all potential structures
    print("Generating annotations for the partition sets:", partitions)
    structures = {}
    best_score = -1e12
    varlabels = True
    for p in partitions:
        # getting a list of block labels to annotated the variables
        block_list = build_partition_block_list(all_symbols, p)
        blocks = len(block_list)

        num_blocks = 1e12
        aggregation_rate = 2
        base_partition = None
        no_improve = 0
        while num_blocks > 10 and no_improve < 10:
            # using the regular expressions to find the variable annotations
            if varlabels:
                if base_partition is None:
                    num_blocks, var_blocks = get_variable_blocks(
                        model_dump, model_dict, variables, p, block_list
                    )

                else:
                    print(
                        "Generating partition with aggregation"
                        f" {aggregation_rate}"
                    )
                    var_blocks = np.empty_like(var_blocks)
                    var_blocks[:] = base_partition

                    num_blocks, var_blocks = merge_adjacent_blocks(
                        var_blocks, aggregation_rate
                    )
                    aggregation_rate += 1

                # getting the equation blocks from the variable blocks
                equ_blocks, merge_cands = get_corresponding_blocks(
                    model_dump, var_blocks, num_blocks, INPUTBLOCKS.VARIABLES
                )

                # if there are merge candidates, then we merge the blocks and then
                # compute the equation labels again
                if False and len(merge_cands) > 0:
                    num_blocks, var_blocks = merge_blocks(
                        var_blocks, merge_cands
                    )
                    equ_blocks, merge_cands = get_corresponding_blocks(
                        model_dump,
                        var_blocks,
                        num_blocks,
                        INPUTBLOCKS.VARIABLES,
                    )

                # performing a refinement of the structure by moving linking
                # variables/equations to blocks
                count = 0
                while count < 100:
                    var_diff, var_blocks = move_linking_to_block(
                        model_dump,
                        var_blocks,
                        equ_blocks,
                        num_blocks,
                        INPUTBLOCKS.EQUATIONS,
                    )

                    equ_diff, equ_blocks = move_linking_to_block(
                        model_dump,
                        equ_blocks,
                        var_blocks,
                        num_blocks,
                        INPUTBLOCKS.VARIABLES,
                    )
                    if var_diff + equ_diff == 0:
                        break

                    count += 1

            else:
                num_blocks, equ_blocks = get_equation_blocks(
                    model_dump, model_dict, equations, p, block_list
                )
                var_blocks, _ = get_corresponding_blocks(
                    model_dump, equ_blocks, num_blocks, INPUTBLOCKS.EQUATIONS
                )
                print(p, np.count_nonzero(equ_blocks == num_blocks + 1))

            # storing the base partition that will be used for the
            # aggregation
            if varlabels and base_partition is None:
                base_partition = np.empty_like(var_blocks)
                base_partition[:] = var_blocks

            structure_score = PipsScore(model_dump, equ_blocks, var_blocks)
            score = structure_score.get_score()

            name = "-".join(p)
            key = f"{name}_{num_blocks}"
            if key in structures.keys():
                continue

            # adding in a limit of no improvement to avoid excessively
            # generating useless structures
            if score <= best_score:
                no_improve += 1
            else:
                no_improve = 0

            if num_blocks > 0 and (not only_best or score > best_score):
                # if we only want the best scoring structure, then we clear the
                # structures dictionary each time the score improves.
                if only_best and score > best_score:
                    best_score = score
                    structures = {}

                structures[key] = {
                    "name": name,
                    "num_blocks": num_blocks,
                    "i_block": equ_blocks,
                    "j_block": var_blocks,
                    "score": structure_score.get_score(),
                }

            print("Score:", key, structure_score.get_score())

            # the aggregation of blocks is not implemented for the equation
            # labels
            if not varlabels:
                break

    # if no structure is found, then we return a single block structure
    if len(structures) == 0:
        structures["single-block"] = {
            "name": "sb",
            "num_blocks": 1,
            "i_block": np.full(len(model_dump["e"].records), 2),
            "j_block": np.full(len(model_dump["x"].records), 2),
            "score": 1,
        }

    return structures


def hierarchical_detection(container):
    pass
