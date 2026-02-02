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
    if (self.blocks.row_perm is not None and self.blocks.col_perm is not None):
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


def plot_coefficient_heatmap(self,
                             figsize=(10, 8),
                             max_display=None,
                             log_scale=True):
    """
    Heatmap showing coefficient magnitudes (log scale)
    Useful for detecting numerical issues
    """
    A_display = self.A

    if max_display:
        max_r, max_c = max_display
        A_display = self.A[:max_r, :max_c].toarray()
    else:
        A_display = self.A.toarray()

    # Take absolute value and add small epsilon to avoid log(0)
    if log_scale:
        A_display = np.log10(np.abs(A_display) + 1e-20)
        A_display[A_display < -10] = np.nan  # Mark zeros as NaN

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A_display,
                   aspect='auto',
                   cmap='viridis',
                   interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax)
    if log_scale:
        cbar.set_label('log₁₀|coefficient|', fontsize=11)
    else:
        cbar.set_label('coefficient magnitude', fontsize=11)

    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Constraints', fontsize=12)
    title = 'Coefficient Magnitude Heatmap'
    if log_scale:
        title += ' (log scale)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig, ax


def plot_row_col_histograms(self, figsize=(12, 5)):
    """
    Distribution of nonzeros per row/column
    Helps identify problem structure
    """
    row_nz = np.diff(self.A.indptr)
    col_nz = np.diff(self.A.tocsc().indptr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Row histogram
    ax1.hist(row_nz, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(row_nz.mean(),
                color='r',
                linestyle='--',
                label=f'Mean: {row_nz.mean():.1f}')
    ax1.set_xlabel('Nonzeros per row', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Row Sparsity Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Column histogram
    ax2.hist(col_nz, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(col_nz.mean(),
                color='r',
                linestyle='--',
                label=f'Mean: {col_nz.mean():.1f}')
    ax2.set_xlabel('Nonzeros per column', fontsize=11)
    ax2.set
