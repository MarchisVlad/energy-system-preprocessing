"""
Academic-standard constraint matrix visualization and analysis
Based on methods from:
- Gondzio & Grothey (2007) on matrix structure
- MIPLIB benchmarking standards
- Bixby (2002) on problem characterization
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mip import Model
from collections import defaultdict

class ConstraintMatrixAnalyser:
    """Academic-standard analyzer for LP/MIP constraint matrices"""
    
    def __init__(self, mps_path=None, A=None, model=None):
        """
        Initialize from either:
        - mps_path: path to MPS file
        - A: pre-built scipy sparse matrix
        - model: python-mip Model object
        """
        if mps_path:
            self.model, self.A = self._read_mps(mps_path)
        elif model:
            self.model = model
            self.A = self._extract_matrix(model)
        elif A is not None:
            self.A = A
            self.model = None
        else:
            raise ValueError("Must provide mps_path, A, or model")
        
        self.A = self.A.tocsr()
        self.n_rows, self.n_cols = self.A.shape
        
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
        
        return sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    
    def get_statistics(self):
        """Compute standard LP/MIP matrix statistics"""
        stats = {
            'dimensions': f'{self.n_rows} × {self.n_cols}',
            'nonzeros': self.A.nnz,
            'density': self.A.nnz / (self.n_rows * self.n_cols),
            'avg_nz_per_row': self.A.nnz / self.n_rows,
            'avg_nz_per_col': self.A.nnz / self.n_cols,
        }
        
        # Row-wise statistics
        row_nz = np.diff(self.A.indptr)
        stats['min_nz_per_row'] = row_nz.min()
        stats['max_nz_per_row'] = row_nz.max()
        stats['std_nz_per_row'] = row_nz.std()
        
        # Column-wise statistics
        col_nz = np.diff(self.A.tocsc().indptr)
        stats['min_nz_per_col'] = col_nz.min()
        stats['max_nz_per_col'] = col_nz.max()
        stats['std_nz_per_col'] = col_nz.std()
        
        # Coefficient statistics
        abs_data = np.abs(self.A.data)
        stats['coeff_min'] = abs_data.min()
        stats['coeff_max'] = abs_data.max()
        stats['coeff_range'] = np.log10(abs_data.max() / abs_data.min()) if abs_data.min() > 0 else np.inf
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        print("=" * 60)
        print("CONSTRAINT MATRIX STATISTICS")
        print("=" * 60)
        print(f"Dimensions:           {stats['dimensions']}")
        print(f"Nonzeros:             {stats['nonzeros']:,}")
        print(f"Density:              {stats['density']:.2%}")
        print(f"\nRow statistics:")
        print(f"  Avg NZ per row:     {stats['avg_nz_per_row']:.2f}")
        print(f"  Min/Max NZ:         {stats['min_nz_per_row']} / {stats['max_nz_per_row']}")
        print(f"  Std dev:            {stats['std_nz_per_row']:.2f}")
        print(f"\nColumn statistics:")
        print(f"  Avg NZ per col:     {stats['avg_nz_per_col']:.2f}")
        print(f"  Min/Max NZ:         {stats['min_nz_per_col']} / {stats['max_nz_per_col']}")
        print(f"  Std dev:            {stats['std_nz_per_col']:.2f}")
        print(f"\nCoefficient range:")
        print(f"  Min/Max:            {stats['coeff_min']:.2e} / {stats['coeff_max']:.2e}")
        print(f"  Log10 range:        {stats['coeff_range']:.2f}")
        print("=" * 60)
    
    def detect_block_structure(self, threshold=0.8):
        """
        Detect block-diagonal or block-angular structure
        Returns block boundaries if found
        """
        # Simple heuristic: look for nearly diagonal blocks
        # More sophisticated: use graph partitioning (METIS, spectral methods)
        
        # Convert to symmetric adjacency matrix for structure detection
        AT = self.A.T
        adjacency = (self.A @ AT) > 0
        
        # TODO: Implement proper block detection (e.g., using networkx or METIS)
        return None
    
    def plot_sparsity_pattern(self, figsize=(10, 8), markersize=1, 
                              show_blocks=False, max_display=None):
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
        ax.spy(A_display, markersize=markersize, aspect='auto')
        
        ax.set_xlabel('Variables (columns)', fontsize=12)
        ax.set_ylabel('Constraints (rows)', fontsize=12)
        ax.set_title('Constraint Matrix Sparsity Pattern', fontsize=14, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        stats = self.get_statistics()
        info_text = (f"Dimensions: {stats['dimensions']}\n"
                    f"Density: {stats['density']:.2%}\n"
                    f"Nonzeros: {stats['nonzeros']:,}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_coefficient_heatmap(self, figsize=(10, 8), max_display=None, 
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
        im = ax.imshow(A_display, aspect='auto', cmap='viridis', 
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
        ax1.axvline(row_nz.mean(), color='r', linestyle='--', 
                   label=f'Mean: {row_nz.mean():.1f}')
        ax1.set_xlabel('Nonzeros per row', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Row Sparsity Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Column histogram
        ax2.hist(col_nz, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(col_nz.mean(), color='r', linestyle='--',
                   label=f'Mean: {col_nz.mean():.1f}')
        ax2.set_xlabel('Nonzeros per column', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Column Sparsity Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def get_sign_pattern(self, max_rows=None, max_cols=None, 
                         canonicalize=False, tol=1e-12):
        """
        Get sign pattern representation
        Returns list of strings like "[ + - 0 + ]"
        """
        A = self.A.copy()
        
        if canonicalize and self.model:
            # Convert to <= form by multiplying >= constraints by -1
            A = A.tolil()
            for i, constr in enumerate(self.model.constrs):
                if hasattr(constr, 'expr') and hasattr(constr.expr, 'sense'):
                    if constr.expr.sense == '>':
                        A.data[i] = [-v for v in A.data[i]]
            A = A.tocsr()
        
        rows_to_show = min(max_rows or self.n_rows, self.n_rows)
        cols_to_show = min(max_cols or self.n_cols, self.n_cols)
        
        sign_strings = []
        for i in range(rows_to_show):
            row = A.getrow(i).toarray().ravel()[:cols_to_show]
            signs = []
            for x in row:
                if abs(x) <= tol:
                    signs.append('0')
                elif x > 0:
                    signs.append('+')
                else:
                    signs.append('-')
            sign_strings.append('[ ' + ' '.join(signs) + ' ]')
        
        return sign_strings
    
    def full_report(self, output_prefix='matrix_analysis'):
        """Generate comprehensive analysis report with all plots"""
        self.print_statistics()
        
        # Generate all plots
        fig1, _ = self.plot_sparsity_pattern(max_display=(500, 500))
        fig1.savefig(f'{output_prefix}_sparsity.png', dpi=150, bbox_inches='tight')
        
        fig2, _ = self.plot_coefficient_heatmap(max_display=(200, 200))
        fig2.savefig(f'{output_prefix}_heatmap.png', dpi=150, bbox_inches='tight')
        
        fig3, _ = self.plot_row_col_histograms()
        fig3.savefig(f'{output_prefix}_histograms.png', dpi=150, bbox_inches='tight')
        
        print(f"\nPlots saved with prefix: {output_prefix}")
        plt.show()