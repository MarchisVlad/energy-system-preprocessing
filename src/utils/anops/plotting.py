from pathlib import Path

import gams.transfer as gt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.sparse import coo_matrix


def get_A_sparse(container):
    A = container["A"].records
    data = np.array(A["value"]) != 0
    row = A["i"].map(lambda x: x.lstrip("e")).astype(int).values - 1
    col = A["j"].map(lambda x: x.lstrip("xbi")).astype(int).values - 1
    return coo_matrix((data, (row, col)))


def createPatches(equ_stage, var_stage, colormap):
    blocks = np.union1d(np.unique(equ_stage), np.unique(var_stage))
    block_patches = []

    if len(blocks) <= 3:
        print("No proper annotation found, skipping block coloration")
        return block_patches

    # Get colormap for blocks
    cmap = colormap
    if type(colormap) is not colors.LinearSegmentedColormap:
        cmap = plt.colormaps.get_cmap(colormap)

    for block in blocks:
        if block == 0:
            block_patches.append(
                patches.Rectangle(
                    (0, 0),
                    np.count_nonzero(var_stage == block),
                    np.count_nonzero(equ_stage),
                    color=(0.0, 0.7, 0.7, 1),
                )
            )
        # if block == 1:
        # block_patches.append(
        # patches.Rectangle(
        # (0, 0),
        # np.count_nonzero(var_stage == block),
        # np.count_nonzero(equ_stage),
        # color=(0.7, 0.7, 0.7, 1),
        # )
        # )

        elif block == blocks.max():
            block_patches.append(
                patches.Rectangle(
                    (0, np.argmax(equ_stage == block)),
                    np.count_nonzero(var_stage >= 0),
                    np.count_nonzero(equ_stage == block),
                    color=(0.7, 0.7, 0.7, 1),
                )
            )

        else:
            block_patches.append(
                patches.Rectangle(
                    (
                        np.argmax(var_stage == block),
                        np.argmax(equ_stage == block),
                    ),
                    np.count_nonzero(var_stage == block),
                    np.count_nonzero(equ_stage == block),
                    color=cmap((block - 1) / (blocks.max() - 1)),
                )
            )
    return block_patches


def plot_gdx(file, plain=False, figure=None, colormap="viridis"):
    file = Path(file)
    cont = gt.Container(file)

    # Get inital equation and variable stages
    equ_stage = cont["e"].records["scale"].values
    var_stage = cont["x"].records["scale"].values

    # the following adds the label 0 to the linking variables. This is to
    # improve the output of the plot. However, the label 0 is not needed for
    # the GDX, since it is not a valid block number
    num_blocks = int(max(np.max(equ_stage), np.max(var_stage)))
    equ_var = (
        cont["A"].records.groupby("i", observed=True)["j"].apply(list).values
    )
    x_to_block = {i: j for i, j in zip(cont["j"].records["uni"], var_stage)}

    # redefining the linking variables and setting them to have the label 0.
    # These variables are those that are present in constraints that contain
    # more than one block variable.
    x_blocks = {}
    for i, vs in enumerate(equ_var):
        if equ_stage[i] == 1 or equ_stage[i] == num_blocks:
            continue

        for v in vs:
            if v in x_blocks:
                if x_blocks[v] != equ_stage[i]:
                    x_to_block[v] = np.int64(0)
            else:
                x_blocks[v] = equ_stage[i]

    # further refinement of the linking variables.
    # these are all the variables that were not labelled initially, but exist
    # in constraints labelled as block constraints.
    # NOTE: in GCG, these constraints would be converted into linking
    # constraints.
    for v, b in x_blocks.items():
        if x_to_block[v] == 1:
            x_to_block[v] = np.int64(0)

    # updating the variable stage with the new labels for the linking variables
    var_stage = np.array(list(x_to_block.values()))

    # Sort by stage while maintaining initial order
    equ_sorted = equ_stage.argsort(kind="mergesort")
    var_sorted = var_stage.argsort(kind="mergesort")

    # Update ordering of stages, names and A matrix
    equ_stage = equ_stage[equ_sorted]
    var_stage = var_stage[var_sorted]
    equ_names = cont["i"].records["element_text"].values[equ_sorted]
    var_names = cont["j"].records["element_text"].values[var_sorted]
    A_coo = get_A_sparse(cont)
    A = A_coo.tocsc()[equ_sorted, :][:, var_sorted]

    # Initialize figure
    fig = plt.figure()
    axes = fig.add_subplot(111, aspect="equal")

    # Add colored block structure
    if plain is False:
        stagePatches = createPatches(equ_stage, var_stage, colormap=colormap)
        for p in stagePatches:
            axes.add_patch(p)

    # Add nonzero elements
    axes.spy(A, markersize=0.25, precision=0, c="k")

    def getEquName(y, pos):
        if y <= len(equ_names):
            return equ_names[int(y)]
        return

    def getVarName(x, pos):
        if x <= len(var_names):
            return var_names[int(x)]
        return

    axes.xaxis.set_major_formatter(ticker.FuncFormatter(getVarName))
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(getEquName))

    plt.xticks(rotation=45, horizontalalignment="left")
    axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer="true"))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer="true"))
    if figure is None:
        plt.show()
    else:
        plt.savefig(figure, bbox_inches="tight", dpi=300)
        print(f"Annotation plot saved to {figure}")
