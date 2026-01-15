import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BlockStructurePlotter:
    """
    Visualization-only tool.
    """

    def plot(self, A, block_structure: BlockStructure,
             figsize=(12, 10), markersize=0.5):

        if block_structure.row_permutation is not None:
            A = A[block_structure.row_permutation]
        if block_structure.col_permutation is not None:
            A = A[:, block_structure.col_permutation]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spy(A, markersize=markersize)

        for r0, h, c0, w in block_structure.boundaries():
            ax.add_patch(
                Rectangle((c0, r0), w, h,
                          linewidth=2, edgecolor='red', facecolor='none')
            )

        ax.set_title(f"Detected blocks: {block_structure.count}")
        ax.set_xlabel("Variables")
        ax.set_ylabel("Constraints")

        return fig, ax
