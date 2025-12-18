# Requires: pip install python-mip scipy matplotlib
from mip import Model  # python-mip
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def read_model_and_get_matrix(mps_path):
    m = Model()
    m.read(mps_path)    # reads an LP/MPS file into the python-mip Model
    # python-mip exposes matrix info via columns/rows and nonzeros counts
    # We'll build a sparse matrix of shape (num_constraints, num_vars)
    n_rows = m.num_rows
    n_cols = m.num_cols
    # python-mip doesn't provide a direct SciPy matrix, so iterate nonzeros
    data = []
    rows = []
    cols = []
    for j in range(n_cols):
        # get coefficients for column j: returns list of (row_index, coeff)
        col = m.col(j)
        if col is None:
            continue
        for (i, coeff) in col:
            rows.append(i)
            cols.append(j)
            data.append(coeff)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return m, A

def coeff_to_sign(x, tol=1e-12):
    """Map a float to '+', '-', or '0' using tolerance tol."""
    if abs(x) <= tol:
        return '0'
    return '+' if x > 0 else '-'

def sign_matrix_strings(A, max_cols=None, tol=1e-12, canonicalize=False, senses=None):
    """
    Produce list of strings, one per constraint row, like "[ + + - 0 0 ]".
    - A: scipy.sparse matrix (rows = constraints, cols = variables)
    - max_cols: if set, truncate to first max_cols columns for display
    - canonicalize: if True, multiply rows by -1 where the corresponding sense is '>='
      (turns constraints to <= canonical form). senses is a list/array of constraint senses
      with values like '<=', '>=', '=' or 'L','G','E' depending on backend.
    """
    A = A.tocsr()
    n_rows, n_cols = A.shape
    if max_cols is None:
        max_cols = n_cols

    # Optionally convert rows to a canonical sense (e.g. all <=) if user supplies senses.
    if canonicalize and senses is not None:
        # make a copy (dense ops on row by row can be easier)
        A = A.copy().tolil()
        for i, s in enumerate(senses):
            if s in ('>=', 'G', 'g'):      # depends on how your backend expresses senses
                A.rows[i] = A.rows[i]   # ensure stored
                A.data[i] = [-v for v in A.data[i]]  # multiply row by -1
        A = A.tocsr()

    rows_strings = []
    for i in range(n_rows):
        row = A.getrow(i).toarray().ravel()[:max_cols]
        signs = [coeff_to_sign(x, tol=tol) for x in row]
        rows_strings.append('[ ' + ' '.join(signs) + ' ]')
    return rows_strings

def plot_sparsity(A, figsize=(6,6), markersize=1):
    plt.figure(figsize=figsize)
    plt.spy(A, markersize=markersize)
    plt.xlabel('variables (columns)')
    plt.ylabel('constraints (rows)')
    plt.title('Constraint matrix sparsity pattern')
    plt.show()

# --- Example usage ---
if __name__ == '__main__':
    model, A = read_model_and_get_matrix('/Users/marchisvlad/energy-system-preprocessing/models/mps/MIPLIB/10teams.mps')
    # if you want to canonicalize: gather constraint senses from model if available
    # Many libraries expose constraint senses by row; python-mip doesn't provide a direct list API
    # in the same shape, so canonicalize=False by default unless you collect senses separately.
    sign_rows = sign_matrix_strings(A, max_cols=50, tol=1e-9, canonicalize=False, senses=None)
    for r in sign_rows[:40]:   # print up to first 40 constraint rows
        print(r)

    # plot sparsity
    plot_sparsity(A)
