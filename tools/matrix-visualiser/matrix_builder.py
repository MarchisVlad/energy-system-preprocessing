# Requires: pip install python-mip scipy matplotlib
from mip import Model  # python-mip
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def read_model_and_get_matrix(mps_path):
    m = Model()
    m.read(mps_path)
    
    n_rows = len(m.constrs)
    n_cols = len(m.vars)
    
    data = []
    rows = []
    cols = []
    
    for i, constr in enumerate(m.constrs):
        expr = constr.expr  # LinExpr
        # In python-mip, LinExpr has attributes: expr (dict), const (float), sense (str)
        # The expr attribute is a dict mapping Var -> coefficient
        for var, coeff in expr.expr.items():
            j = var.idx  # Use var.idx instead of var.index
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
    model, A = read_model_and_get_matrix('/homes/vm922/energy-system-preprocessing/models/mps/MIPLIB/10teams.mps')
    
    # if you want to canonicalize: gather constraint senses from model if available
    # You can extract senses like this:
    # senses = [c.expr.sense for c in model.constrs]
    
    sign_rows = sign_matrix_strings(A, max_cols=50, tol=1e-9, canonicalize=False, senses=None)
    
    for r in sign_rows[:40]:   # print up to first 40 constraint rows
        print(r)
    
    # plot sparsity
    plot_sparsity(A)