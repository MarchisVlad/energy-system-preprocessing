from enum import Enum
import src.core
from src.detection import RCMDetection, SpectralDetection
import mip
from typing import Union
import gamspy as gp


class ProblemType(Enum):
    MIN = 1
    MAX = 2


class ProblemClass(Enum):
    LP = 1
    MIP = 2
    QP = 3


class ModelFormat(Enum):
    MPS = 1
    GMS = 2
    GDX = 3
    LP = 4


class Model:

    model = Union[gp.Model, mip.Model]
    problem_type: ProblemType = None
    problem_class: ProblemClass = None
    model_format: ModelFormat = None
    blocks: BlockStructure = None
    presolves: List[PresolvingMethod] = None  # e.g. "CoeffTightening"

    def __init__(
        self,
        A,
        problem_type=ProblemType.MIN,
        problem_class=ProblemClass.MIP,
        model_format=ModelFormat.MPS,
        blocks=None,
        presolves=[],
    ):

        self.A = A
        self.problem_type = problem_type
        self.problem_class = problem_class
        self.model_format = model_format
        self.blocks = blocks

        self.presolves = presolves

    def __post_init__(self):
        if (isinstance(self.model, mip.Model) and
                not self.model_format is ModelFormat.MPS):
            raise TypeError(
                "Models must have their format specified: attempted to" +
                " construct a mip.Model without setting the model_format parameter."
            )
        if (isinstance(self.model, gp.Model) and
                not self.model_format is ModelFormat.GMS):
            raise Type

    def __init__(self, path, model_format=ModelFormat.MPS):
        pass

    def _read_mps(self, mps_path):
        """Read MPS file using python-mip"""
        m = Model()
        m.read(mps_path)
        A = self._extract_matrix(m)
        return m, A

    def _extract_matrix(self, model):
        """Extract constraint matrix from python-mip Model"""
        n_rows = len(model.constrs)
        n_cols = len(model.vars)

        data, rows, cols = [], [], []

        for i, constr in enumerate(model.constrs):
            expr = constr.expr
            for var, coeff in expr.expr.items():
                rows.append(i)
                cols.append(var.idx)
                data.append(coeff)

        return sp.coo_matrix((data, (rows, cols)),
                             shape=(n_rows, n_cols)).tocsr()

    def _get_col_ordering(self, row_perm):
        """Get column ordering that follows row ordering"""
        # Simple heuristic: order columns by first row they appear in
        A_reordered = self.A[row_perm]

        col_first_row = np.zeros(self.n_cols, dtype=int)
        for col in range(self.n_cols):
            col_data = A_reordered.getcol(col)
            rows = col_data.nonzero()[0]
            if len(rows) > 0:
                col_first_row[col] = rows[0]
            else:
                col_first_row[col] = self.n_rows  # Put empty columns at end

        col_perm = np.argsort(col_first_row)
        return col_perm

    def _choose_detection_method(self):
        """Choose best detection method based on matrix properties"""
        # For small matrices, use spectral
        if self.n_rows < 1000:
            return RCMDetection()
        # For very sparse matrices, try pattern recognition first
        elif self.A.nnz / (self.n_rows * self.n_cols) < 0.01:
            return "pattern"
        # Default to RCM for medium/large matrices
        else:
            return "rcm"

    def plot_sparsity_pattern(self,
                              figsize=(10, 8),
                              markersize=1,
                              show_blocks=False,
                              max_display=None):
        """
        Standard sparsity pattern visualization

        Args:
            max_display: tuple (max_rows, max_cols) to limit display size
        """
        A_display = self.A

        if max_display:
            max_r, max_c = max_display
            A_display = self.A[:max_r, :max_c]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spy(A_display, markersize=markersize, aspect="auto")

        ax.set_xlabel("Variables (columns)", fontsize=12)
        ax.set_ylabel("Constraints (rows)", fontsize=12)
        ax.set_title("Constraint Matrix Sparsity Pattern",
                     fontsize=14,
                     fontweight="bold")

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--")

        stats = self.get_statistics()
        info_text = (f"Dimensions: {stats['dimensions']}\n"
                     f"Density: {stats['density']:.2%}\n"
                     f"Nonzeros: {stats['nonzeros']:,}")
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

        plt.tight_layout()
        return fig, ax

    def plot_block_structure(self,
                             block_structure=None,
                             figsize=(12, 10),
                             markersize=0.5):
        """Visualize detected block structure"""
        if self.blocks is None:
            self.blocks = self.detect_block_structure()

        if self.blocks is None:
            print("No block structure detected")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Reorder matrix according to block structure
        if (self.blocks.row_perm is not None and
                self.blocks.col_perm is not None):
            A_reordered = self.A[self.blocks.row_perm][:, self.blocks.col_perm]
        else:
            A_reordered = self.A

        # Plot sparsity pattern
        ax.spy(A_reordered, markersize=markersize, aspect="auto")

        # Draw block boundaries
        boundaries = self.blocks.get_boundaries()
        for boundary in boundaries:
            rect = Rectangle(
                (boundary["col_start"], boundary["row_start"]),
                boundary["col_width"],
                boundary["row_height"],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title("Block Structure Visualization",
                     fontsize=14,
                     fontweight="bold")
        ax.set_xlabel("Variables (reordered)", fontsize=12)
        ax.set_ylabel("Constraints (reordered)", fontsize=12)

        # Add block statistics
        stats_text = f"Blocks detected: {self.blocks.n_blocks}\n"
        stats_text += f"Method: {self.blocks.method}"
        if self.blocks.pattern_type:
            stats_text += f"\nPattern: {self.blocks.pattern_type}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            fontsize=9,
        )

        plt.tight_layout()
        return fig, ax
