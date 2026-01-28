import os
from pathlib import Path

import gams.transfer as gt
import numpy as np
import pandas as pd


from anops.detection import frequency_detection, equation_blocks_from_variables

os.environ["GDXCOMPRESS"] = "1"


def find_set_elements(container, pattern_regex, sort_order=None) -> np.array:
    set_df = (
        container["j"]
        .records["element_text"]
        .str.extract(pattern_regex)
        .dropna()
    )
    set_unique = pd.Series(
        pd.Series(zip(*[set_df[col] for col in set_df.columns])).unique()
    )

    if sort_order is not None:
        # Extract numeric values for each column and use sort_order for obtaining the sorting index
        set_columns = pd.DataFrame(set_unique.to_list())
        set_numeric = pd.concat(
            [
                set_columns[col]
                .str.extract(r"(\d+)")
                .astype(int)
                .rename(columns={0: i})
                for i, col in enumerate(set_columns.columns)
            ],
            axis=1,
        )
        set_unique = set_unique.loc[
            set_numeric.sort_values(sort_order, kind="mergesort").index
        ].reset_index(drop=True)

    if len(set_unique) == 0:
        raise ValueError(f"No set elements found using pattern {pattern_regex}")

    return set_unique.values


def get_pattern_variable_blocks(
    container,
    pattern_regex,
    pattern_to_block=None,
    sort_order=None,
    blocks=None,
):
    set_elements = find_set_elements(
        container, pattern_regex, sort_order=sort_order
    )
    if len(set_elements[0]) == 1:
        print(
            f"Found {len(set_elements)} unique set elements for pattern {pattern_regex}"
        )
    else:
        print(
            f"Found {len(set_elements)} unique set combinations for pattern {pattern_regex}"
        )

    if pattern_to_block is None:
        # If no blocks are specificed, use each unique element
        if blocks is None:
            blocks = len(set_elements)

        # Generate dictionary of assignments from elements to blocks
        element_to_block = {
            tuple(set_element): block_num + 2
            for block_num, set_list in enumerate(
                np.array_split(set_elements, blocks)
            )
            for set_element in set_list
        }

    else:
        # Ensure the keys are tuples
        if not isinstance(list(pattern_to_block)[0], tuple):
            pattern_to_block = {(i,): j for i, j in pattern_to_block.items()}

        # Ensure the lowest mapped block is 2
        min_block = min(pattern_to_block.values())
        if min_block < 2:
            pattern_to_block = {
                i: j - min_block + 2 for i, j in pattern_to_block.items()
            }

        element_to_block = pattern_to_block

    # Get variable annotation
    j_set_groups = (
        container["j"]
        .records["element_text"]
        .str.extract(pattern_regex)
        .values.astype(str)
    )
    j_set = [tuple(elements) for elements in j_set_groups]

    j_blocks = np.array([element_to_block.get(j, 1) for j in j_set]).astype(
        float
    )
    blocks = len(set(np.unique(j_blocks)) - {1})
    return blocks, j_blocks


def identify_structure_and_annotate_gdx(
    model_dump_file,
    model_dict_file=None,
    all_symbols_file=None,
    pattern_regex=None,
    pattern_to_block=None,
    sort_order=None,
    blocks=None,
    suffix="_annot",
    num_partition_sets=5,
):
    """
    identifies structure in the model, either based on the supplied pattern or
    detected from the model symbols, then generates an annotated GDX file.

    Detection is triggered if pattern is None. A model_dict_file and
    all_symbols_file is required for detection. When detecting structure,
    multiple GDX files are generated---one for each structure detected.

    When using the pattern, the all_symbols_file is not used. Only a single GDX
    file is generated
    """
    file = Path(model_dump_file)

    # s1 = time.perf_counter()
    container = gt.Container(model_dump_file)
    # e1 = time.perf_counter()
    # print(f"Container in {(e1-s1):.2f} seconds")

    # if the pattern is None, then structure detection is performed. Otherwise,
    # the pattern is used to find the variable blocks
    if pattern_regex is not None:
        blocks, j_block = get_pattern_variable_blocks(
            container=container,
            pattern_regex=pattern_regex,
            pattern_to_block=pattern_to_block,
            sort_order=sort_order,
            blocks=blocks,
        )
        i_block, _ = equation_blocks_from_variables(container, j_block, blocks)

        annotate_gdx(container, i_block, j_block, blocks, file, suffix)
    else:
        if model_dict_file is None:
            raise IOError(
                "A model dictionary file is needed for automatic detection"
            )

        if all_symbols_file is None:
            raise IOError("A symbols file is needed for automatic detection")

        model_dict = gt.Container(model_dict_file)
        all_symbols = gt.Container(all_symbols_file)
        detected_structures = frequency_detection(
            container, model_dict, all_symbols, num_partition_sets
        )

        annotated_instances = file.parent.joinpath(f"{file.stem}{suffix}.inst")
        with open(annotated_instances, "w") as f:
            pass

        for k, structure in detected_structures.items():
            annotate_gdx(
                container,
                structure["i_block"],
                structure["j_block"],
                structure["num_blocks"],
                file,
                "{}_{}".format(suffix, structure["name"]),
                annotated_instances,
            )


def annotate_gdx(container, i_block, j_block, blocks, file, suffix="",
                 annotated_instances=None):
    # Update records and write out new gdx
    container["x"].records["scale"] = j_block
    container["e"].records["scale"] = i_block
    file_out = file.parent.joinpath(f"{file.stem}{suffix}_{blocks}b.gdx")

    container.write(file_out.as_posix())

    if annotated_instances is not None:
       with open(annotated_instances, "a") as f:
           f.write(f"{file_out} {blocks}\n")

    print(f"Finished annotating file {file_out}")
