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

class BlockStructure:
    """Container for block structure information"""
    
    def __init__(self, row_partition=None, col_partition=None, 
                 blocks=None, row_perm=None, col_perm=None,
                 pattern_type=None, pattern_info=None, method='auto'):
        self.row_partition = row_partition
        self.col_partition = col_partition
        self.blocks = blocks
        self.row_perm = row_perm
        self.col_perm = col_perm
        self.pattern_type = pattern_type
        self.pattern_info = pattern_info
        self.method = method
        
        if row_partition is not None:
            self.n_blocks = len(np.unique(row_partition))
        elif blocks is not None:
            self.n_blocks = len(blocks)
        else:
            self.n_blocks = 0
    
    def get_boundaries(self):
        """Get block boundary coordinates for visualization"""
        if self.blocks is None:
            return []
        
        boundaries = []
        for block in self.blocks:
            boundaries.append({
                'row_start': block['row_start'],
                'row_height': block['row_end'] - block['row_start'],
                'col_start': block['col_start'],
                'col_width': block['col_end'] - block['col_start']
            })
        return boundaries
    
    def summary(self):
        """Print summary of block structure"""
        print(f"\nBlock Structure Summary:")
        print(f"  Detection method: {self.method}")
        print(f"  Number of blocks: {self.n_blocks}")
        if self.pattern_type:
            print(f"  Primary pattern type: {self.pattern_type}")
        
        # Show all detected patterns if available
        if self.pattern_info and 'all_detected_patterns' in self.pattern_info:
            all_patterns = self.pattern_info['all_detected_patterns']
            print(f"  All detected patterns: {list(all_patterns.keys())}")
        
        # Pattern-specific details
        if self.pattern_info:
            if 'bandwidth' in self.pattern_info:
                print(f"  Bandwidth: {self.pattern_info['bandwidth']}")
            if 'linking_rows' in self.pattern_info:
                n_linking = self.pattern_info.get('n_linking', 0)
                print(f"  Linking constraints: {n_linking}")
            if 'type' in self.pattern_info:  # For staircase
                print(f"  Structure type: {self.pattern_info['type']}")
        
        if self.blocks:
            print(f"  Block details:")
            for i, block in enumerate(self.blocks):
                rows_size = block['row_end'] - block['row_start']
                cols_size = block['col_end'] - block['col_start']
                print(f"    Block {i}: rows [{block['row_start']}:{block['row_end']}] ({rows_size}), "
                      f"cols [{block['col_start']}:{block['col_end']}] ({cols_size})")


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
    
    def detect_block_structure(self, method='auto', n_blocks=None, 
                              min_block_size=10, threshold=0.1):
        """
        Detect block structure in constraint matrix
        
        Args:
            method: 'metis', 'spectral', 'rcm', 'pattern', or 'auto'
            n_blocks: Number of blocks (if known)
            min_block_size: Minimum rows/cols per block
            threshold: Density threshold for considering blocks separate
        
        Returns:
            BlockStructure object with partition info and reordering
        """
        if method == 'auto':
            method = self._choose_detection_method()
        
        if method == 'metis':
            return self._detect_metis(n_blocks)
        elif method == 'spectral':
            return self._detect_spectral(n_blocks)
        elif method == 'rcm':
            return self._detect_rcm()
        elif method == 'pattern':
            return self._detect_patterns(threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _choose_detection_method(self):
        """Choose best detection method based on matrix properties"""
        # For small matrices, use spectral
        if self.n_rows < 1000:
            return 'spectral'
        # For very sparse matrices, try pattern recognition first
        elif self.A.nnz / (self.n_rows * self.n_cols) < 0.01:
            return 'pattern'
        # Default to RCM for medium/large matrices
        else:
            return 'rcm'
    
    def _detect_metis(self, n_blocks):
        """Use METIS-style graph partitioning"""
        try:
            import networkx as nx
            from networkx.algorithms.community import greedy_modularity_communities
        except ImportError:
            print("Warning: networkx required for METIS-style partitioning, falling back to RCM")
            return self._detect_rcm()
        
        # Build constraint-variable bipartite graph
        col_to_rows = defaultdict(set)
        rows, cols = self.A.nonzero()
        
        for r, c in zip(rows, cols):
            col_to_rows[c].add(r)
        
        # Connect constraints that share variables
        constraint_graph = nx.Graph()
        for c, row_set in col_to_rows.items():
            row_list = list(row_set)
            for i in range(len(row_list)):
                for j in range(i+1, len(row_list)):
                    if constraint_graph.has_edge(row_list[i], row_list[j]):
                        constraint_graph[row_list[i]][row_list[j]]['weight'] += 1
                    else:
                        constraint_graph.add_edge(row_list[i], row_list[j], weight=1)
        
        # Partition using community detection
        if len(constraint_graph.nodes()) == 0:
            return None
        
        communities = list(greedy_modularity_communities(constraint_graph, weight='weight'))
        
        row_partition = np.zeros(self.n_rows, dtype=int)
        for block_id, community in enumerate(communities):
            for row in community:
                row_partition[row] = block_id
        
        col_partition = self._compute_col_partition(row_partition)
        row_perm, col_perm = self._partitions_to_permutations(row_partition, col_partition)
        blocks = self._extract_blocks_from_partitions(row_partition, col_partition, row_perm, col_perm)
        
        return BlockStructure(row_partition, col_partition, blocks, row_perm, col_perm, method='metis')
    
    def _detect_spectral(self, n_blocks):
        """Spectral clustering on constraint connectivity"""
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            print("Warning: sklearn required for spectral clustering, falling back to RCM")
            return self._detect_rcm()
        
        # Build constraint similarity matrix
        AT = self.A.T
        similarity = (self.A @ AT).astype(float)
        
        if n_blocks is None:
            n_blocks = self._estimate_n_blocks(similarity)
        
        # Always convert to dense and use precomputed affinity
        # This avoids the API warning about constructing affinity from data
        similarity_matrix = similarity.toarray()
        
        # Make sure the matrix is symmetric and non-negative
        similarity_matrix = np.maximum(similarity_matrix, 0)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        
        try:
            clustering = SpectralClustering(n_clusters=n_blocks, 
                                          affinity='precomputed',
                                          random_state=42,
                                          assign_labels='kmeans')
            row_partition = clustering.fit_predict(similarity_matrix)
        except Exception as e:
            # Fallback if spectral fails
            print(f"Warning: Spectral clustering failed ({str(e)}), falling back to RCM")
            return self._detect_rcm()
        
        col_partition = self._compute_col_partition(row_partition)
        row_perm, col_perm = self._partitions_to_permutations(row_partition, col_partition)
        blocks = self._extract_blocks_from_partitions(row_partition, col_partition, row_perm, col_perm)
        
        return BlockStructure(row_partition, col_partition, blocks, row_perm, col_perm, method='spectral')
    
    def _detect_rcm(self):
        """Reverse Cuthill-McKee reordering to reveal structure"""
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        
        # Build symmetric structure matrix
        AT = self.A.T
        structure = self.A @ AT
        structure = (structure != 0).astype(int)
        
        # Get RCM permutation
        try:
            row_perm = reverse_cuthill_mckee(structure, symmetric_mode=True)
        except:
            row_perm = np.arange(self.n_rows)
        
        col_perm = self._get_col_ordering(row_perm)
        
        # Detect blocks in reordered matrix
        A_reordered = self.A[row_perm][:, col_perm]
        blocks = self._find_blocks_in_ordered_matrix(A_reordered, row_perm, col_perm)
        
        return BlockStructure(blocks=blocks, row_perm=row_perm, col_perm=col_perm, method='rcm')
    
    def _detect_patterns(self, threshold):
        """Detect specific patterns: block-diagonal, block-angular, staircase"""
        patterns = {
            'block_diagonal': self._check_block_diagonal(threshold),
            'block_angular': self._check_block_angular(threshold),
            'staircase': self._check_staircase(threshold),
            'bordered': self._check_bordered(threshold)
        }
        
        # Collect all detected patterns
        detected = [(name, result) for name, result in patterns.items() if result is not None]
        
        if len(detected) > 1:
            # Multiple patterns detected - choose the best one or combine info
            print(f"Note: Multiple patterns detected: {[name for name, _ in detected]}")
            # Score each pattern by quality of blocks
            best_pattern = self._choose_best_pattern(detected)
            pattern_name, pattern_info = best_pattern
            
            # Store info about all detected patterns
            all_patterns = {name: info for name, info in detected}
            pattern_info['all_detected_patterns'] = all_patterns
            
        elif len(detected) == 1:
            pattern_name, pattern_info = detected[0]
        else:
            # No patterns detected
            pattern_name = None
            pattern_info = None
        
        if pattern_name:
            blocks = pattern_info.get('blocks')
            
            # If no blocks found in pattern, try to create simple blocks
            if blocks is None:
                blocks = self._create_simple_blocks()
            
            return BlockStructure(pattern_type=pattern_name, 
                                pattern_info=pattern_info,
                                blocks=blocks,
                                method='pattern')
        
        # If no patterns detected, try to find ANY structure
        simple_blocks = self._create_simple_blocks()
        if simple_blocks and len(simple_blocks) > 1:
            return BlockStructure(blocks=simple_blocks, 
                                pattern_type='generic',
                                method='pattern')
        
        return None
    
    def _choose_best_pattern(self, detected_patterns):
        """
        Choose the best pattern when multiple are detected
        Priority: block_diagonal > block_angular > staircase > bordered
        Also considers quality metrics like number of blocks
        """
        # Score each pattern
        scores = []
        for name, info in detected_patterns:
            score = 0
            
            # Priority weights
            priority = {
                'block_diagonal': 100,
                'block_angular': 80,
                'staircase': 60,
                'bordered': 40
            }
            score += priority.get(name, 0)
            
            # Bonus for having actual blocks
            if info.get('blocks'):
                n_blocks = len(info['blocks'])
                score += min(n_blocks * 10, 50)  # Cap at 50 bonus points
            
            # Bonus for good structural properties
            if name == 'block_diagonal' and 'bandwidth' in info:
                # Lower bandwidth is better
                bandwidth_ratio = info['bandwidth'] / max(self.n_rows, self.n_cols)
                score += (1 - bandwidth_ratio) * 20
            
            if name == 'block_angular' and 'linking_ratio' in info:
                # Lower linking ratio is better (more independent blocks)
                linking_ratio = info['linking_ratio']
                score += (1 - linking_ratio) * 30
            
            scores.append((score, name, info))
        
        # Return highest scoring pattern
        scores.sort(reverse=True)
        return (scores[0][1], scores[0][2])
    
    def detect_all_patterns(self, threshold=0.1):
        """
        Detect ALL patterns present in the matrix
        Returns a dictionary of all detected patterns
        """
        patterns = {
            'block_diagonal': self._check_block_diagonal(threshold),
            'block_angular': self._check_block_angular(threshold),
            'staircase': self._check_staircase(threshold),
            'bordered': self._check_bordered(threshold)
        }
        
        # Filter out None results
        detected = {name: result for name, result in patterns.items() if result is not None}
        
        return detected if detected else None
    
    def _estimate_n_blocks(self, similarity_matrix):
        """Estimate number of blocks using eigenvalue gap"""
        try:
            from scipy.sparse.linalg import eigsh
            # Compute first few eigenvalues
            n_eigs = min(20, self.n_rows - 2)
            eigenvalues = eigsh(similarity_matrix, k=n_eigs, return_eigenvectors=False)
            
            # Look for largest gap
            gaps = np.diff(sorted(eigenvalues))
            n_blocks = np.argmax(gaps) + 2  # +2 because of diff and 0-indexing
            
            return max(2, min(n_blocks, 10))  # Clamp between 2 and 10
        except:
            return 3  # Default
    
    def _compute_col_partition(self, row_partition):
        """Assign columns to blocks based on row partition"""
        n_blocks = row_partition.max() + 1
        col_partition = np.zeros(self.n_cols, dtype=int)
        
        for col in range(self.n_cols):
            col_data = self.A.getcol(col)
            rows = col_data.nonzero()[0]
            
            if len(rows) > 0:
                # Assign to most common block
                block_counts = np.bincount(row_partition[rows], minlength=n_blocks)
                col_partition[col] = block_counts.argmax()
            else:
                # For empty columns, assign to block 0 (or distribute evenly)
                col_partition[col] = col % n_blocks  # Distribute empty columns
        
        # Ensure each block has at least one column if possible
        for b in range(n_blocks):
            if not np.any(col_partition == b):
                # Find columns that touch this block's rows
                block_rows = np.where(row_partition == b)[0]
                if len(block_rows) > 0:
                    for row in block_rows:
                        row_data = self.A.getrow(row)
                        cols = row_data.nonzero()[1]
                        if len(cols) > 0:
                            # Reassign first column to this block
                            col_partition[cols[0]] = b
                            break
        
        return col_partition
    
    def _partitions_to_permutations(self, row_partition, col_partition):
        """Convert partitions to permutation vectors"""
        # Sort rows by partition
        row_perm = np.argsort(row_partition)
        col_perm = np.argsort(col_partition)
        return row_perm, col_perm
    
    def _extract_blocks_from_partitions(self, row_partition, col_partition, row_perm, col_perm):
        """Extract block boundaries from partitions"""
        n_blocks = row_partition.max() + 1
        blocks = []
        
        for b in range(n_blocks):
            row_mask = row_partition[row_perm] == b
            col_mask = col_partition[col_perm] == b
            
            row_indices = np.where(row_mask)[0]
            col_indices = np.where(col_mask)[0]
            
            if len(row_indices) > 0 and len(col_indices) > 0:
                blocks.append({
                    'row_start': row_indices[0],
                    'row_end': row_indices[-1] + 1,
                    'col_start': col_indices[0],
                    'col_end': col_indices[-1] + 1,
                    'block_id': b
                })
        
        return blocks
    
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
    
    def _find_blocks_in_ordered_matrix(self, A_reordered, row_perm, col_perm):
        """Find diagonal blocks in reordered matrix"""
        # Use a simple sliding window approach
        blocks = []
        block_size = max(10, min(100, self.n_rows // 10))
        
        i = 0
        while i < self.n_rows:
            # Find extent of nonzeros in this block
            block_rows = slice(i, min(i + block_size, self.n_rows))
            block_data = A_reordered[block_rows]
            
            if block_data.nnz > 0:
                cols_in_block = block_data.nonzero()[1]
                if len(cols_in_block) > 0:
                    col_start = cols_in_block.min()
                    col_end = cols_in_block.max() + 1
                    
                    blocks.append({
                        'row_start': i,
                        'row_end': min(i + block_size, self.n_rows),
                        'col_start': int(col_start),
                        'col_end': int(col_end),
                        'block_id': len(blocks)
                    })
            
            i += block_size
        
        return blocks if len(blocks) > 1 else None
    
    def _create_simple_blocks(self):
        """Create simple block partition based on matrix structure"""
        # Partition matrix into roughly equal blocks based on density patterns
        n_candidate_blocks = max(2, min(10, self.n_rows // 50))
        
        if self.n_rows < 20:
            return None
        
        block_size = self.n_rows // n_candidate_blocks
        blocks = []
        
        for i in range(n_candidate_blocks):
            row_start = i * block_size
            row_end = (i + 1) * block_size if i < n_candidate_blocks - 1 else self.n_rows
            
            # Find column extent for this row block
            block_rows = self.A[row_start:row_end]
            if block_rows.nnz > 0:
                col_indices = block_rows.nonzero()[1]
                col_start = int(col_indices.min())
                col_end = int(col_indices.max() + 1)
                
                blocks.append({
                    'row_start': row_start,
                    'row_end': row_end,
                    'col_start': col_start,
                    'col_end': col_end,
                    'block_id': i
                })
        
        return blocks if len(blocks) > 1 else None
    
    def _check_block_diagonal(self, threshold):
        """Check if matrix is approximately block diagonal"""
        # Compute bandwidth
        rows, cols = self.A.nonzero()
        if len(rows) == 0:
            return None
        
        # Calculate how close entries are to the diagonal
        distances = np.abs(rows - cols * (self.n_rows / self.n_cols))
        avg_distance = np.mean(distances)
        max_dimension = max(self.n_rows, self.n_cols)
        
        # If entries cluster near diagonal, likely block diagonal
        if avg_distance < 0.3 * max_dimension:
            # Try to extract diagonal blocks
            blocks = self._extract_diagonal_blocks_improved()
            if blocks and len(blocks) > 1:
                bandwidth = int(np.max(distances))
                return {'blocks': blocks, 'bandwidth': bandwidth, 'avg_distance': avg_distance}
        
        return None
    
    def _extract_diagonal_blocks_improved(self):
        """Improved diagonal block extraction"""
        if self.n_rows == 0 or self.n_cols == 0:
            return None
        
        # Use row-wise approach: group consecutive rows that share column regions
        blocks = []
        current_block_start = 0
        
        # Get column range for each row
        row_col_ranges = []
        for i in range(self.n_rows):
            row = self.A.getrow(i)
            if row.nnz > 0:
                cols = row.nonzero()[1]
                row_col_ranges.append((cols.min(), cols.max()))
            else:
                row_col_ranges.append((None, None))
        
        # Find natural breaks where column ranges shift significantly
        i = 0
        while i < self.n_rows:
            if row_col_ranges[i][0] is None:
                i += 1
                continue
            
            # Start new block
            block_row_start = i
            block_col_min = row_col_ranges[i][0]
            block_col_max = row_col_ranges[i][1]
            
            # Extend block while column ranges overlap significantly
            j = i + 1
            while j < self.n_rows:
                if row_col_ranges[j][0] is None:
                    j += 1
                    continue
                
                # Check if this row's columns overlap with current block
                overlap = min(block_col_max, row_col_ranges[j][1]) - max(block_col_min, row_col_ranges[j][0])
                block_width = block_col_max - block_col_min + 1
                
                if overlap > 0.3 * block_width:  # 30% overlap threshold
                    # Extend block
                    block_col_min = min(block_col_min, row_col_ranges[j][0])
                    block_col_max = max(block_col_max, row_col_ranges[j][1])
                    j += 1
                else:
                    # End of block
                    break
            
            # Save block if it has reasonable size
            if j - block_row_start >= 5:
                blocks.append({
                    'row_start': block_row_start,
                    'row_end': j,
                    'col_start': int(block_col_min),
                    'col_end': int(block_col_max + 1),
                    'block_id': len(blocks)
                })
            
            i = j if j > i else i + 1
        
        return blocks if len(blocks) > 1 else None
    
    def _check_block_angular(self, threshold):
        """Check for block-angular structure"""
        row_nz = np.diff(self.A.indptr)
        
        if len(row_nz) == 0:
            return None
        
        # Look for linking rows (rows with many more nonzeros than average)
        mean_nz = np.mean(row_nz)
        std_nz = np.std(row_nz)
        
        # More flexible threshold: rows with nz > mean + 2*std
        linking_threshold = max(mean_nz + 2 * std_nz, 2 * mean_nz)
        linking_rows = np.where(row_nz > linking_threshold)[0]
        
        # Block angular if we have 1-20% linking rows
        linking_ratio = len(linking_rows) / self.n_rows if self.n_rows > 0 else 0
        
        if 0.01 < linking_ratio < 0.2 and len(linking_rows) > 0:
            # Try to partition non-linking rows into blocks
            non_linking_rows = np.setdiff1d(np.arange(self.n_rows), linking_rows)
            
            if len(non_linking_rows) > 10:
                # Create blocks from non-linking rows
                blocks = self._partition_non_linking_rows(non_linking_rows, linking_rows)
                
                return {
                    'linking_rows': linking_rows.tolist(), 
                    'n_linking': len(linking_rows),
                    'linking_ratio': linking_ratio,
                    'blocks': blocks
                }
        
        return None
    
    def _partition_non_linking_rows(self, non_linking_rows, linking_rows):
        """Partition non-linking rows into blocks"""
        if len(non_linking_rows) == 0:
            return None
        
        # Group non-linking rows by their column usage
        blocks = []
        sorted_rows = sorted(non_linking_rows)
        
        i = 0
        while i < len(sorted_rows):
            block_start_row = sorted_rows[i]
            
            # Get column range for this block
            row_idx = block_start_row
            row_data = self.A.getrow(row_idx)
            if row_data.nnz > 0:
                block_cols = set(row_data.nonzero()[1])
            else:
                i += 1
                continue
            
            # Extend block
            j = i + 1
            block_end_row = sorted_rows[i]
            
            while j < len(sorted_rows) and j - i < 50:  # Limit block size
                row_idx = sorted_rows[j]
                row_data = self.A.getrow(row_idx)
                if row_data.nnz > 0:
                    row_cols = set(row_data.nonzero()[1])
                    # Check overlap
                    overlap = len(block_cols & row_cols)
                    if overlap > 0.2 * len(block_cols):
                        block_cols.update(row_cols)
                        block_end_row = sorted_rows[j]
                        j += 1
                    else:
                        break
                else:
                    j += 1
            
            # Create block
            if len(block_cols) > 0:
                blocks.append({
                    'row_start': block_start_row,
                    'row_end': block_end_row + 1,
                    'col_start': int(min(block_cols)),
                    'col_end': int(max(block_cols) + 1),
                    'block_id': len(blocks)
                })
            
            i = j
        
        return blocks if len(blocks) > 1 else None
    
    def _check_staircase(self, threshold):
        """Check for staircase structure"""
        # Look for predominantly lower-triangular or upper-triangular structure
        rows, cols = self.A.nonzero()
        
        if len(rows) == 0:
            return None
        
        lower_triangular = np.sum(rows >= cols) / len(rows)
        upper_triangular = np.sum(rows <= cols) / len(rows)
        
        if lower_triangular > 0.8:
            return {'type': 'lower_triangular', 'ratio': lower_triangular}
        elif upper_triangular > 0.8:
            return {'type': 'upper_triangular', 'ratio': upper_triangular}
        
        return None
    
    def _check_bordered(self, threshold):
        """Check for bordered block-diagonal structure"""
        # Check if first/last rows and columns are dense
        first_row_nz = np.diff(self.A[0:1].tocsr().indptr)[0]
        last_row_nz = np.diff(self.A[-1:].tocsr().indptr)[0]
        
        avg_row_nz = self.A.nnz / self.n_rows
        
        if first_row_nz > 3 * avg_row_nz or last_row_nz > 3 * avg_row_nz:
            return {'border_rows': [0] if first_row_nz > 3 * avg_row_nz else [] + 
                                   [self.n_rows - 1] if last_row_nz > 3 * avg_row_nz else []}
        
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
    
    def plot_block_structure(self, block_structure=None, figsize=(12, 10), markersize=0.5):
        """Visualize detected block structure"""
        if block_structure is None:
            block_structure = self.detect_block_structure()
        
        if block_structure is None:
            print("No block structure detected")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Reorder matrix according to block structure
        if block_structure.row_perm is not None and block_structure.col_perm is not None:
            A_reordered = self.A[block_structure.row_perm][:, block_structure.col_perm]
        else:
            A_reordered = self.A
        
        # Plot sparsity pattern
        ax.spy(A_reordered, markersize=markersize, aspect='auto')
        
        # Draw block boundaries
        boundaries = block_structure.get_boundaries()
        for boundary in boundaries:
            rect = Rectangle((boundary['col_start'], boundary['row_start']),
                            boundary['col_width'], boundary['row_height'],
                            linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title('Block Structure Visualization', fontsize=14, fontweight='bold')
        ax.set_xlabel('Variables (reordered)', fontsize=12)
        ax.set_ylabel('Constraints (reordered)', fontsize=12)
        
        # Add block statistics
        stats_text = f"Blocks detected: {block_structure.n_blocks}\n"
        stats_text += f"Method: {block_structure.method}"
        if block_structure.pattern_type:
            stats_text += f"\nPattern: {block_structure.pattern_type}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.7), fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_pattern(self, pattern_name, threshold=0.1, figsize=(12, 10), markersize=0.5):
        """
        Plot a specific pattern by name
        
        Args:
            pattern_name: 'block_diagonal', 'block_angular', 'staircase', 'bordered'
            threshold: detection threshold
        """
        # Map pattern names to check methods
        pattern_checks = {
            'block_diagonal': self._check_block_diagonal,
            'block_angular': self._check_block_angular,
            'staircase': self._check_staircase,
            'bordered': self._check_bordered
        }
        
        if pattern_name not in pattern_checks:
            print(f"Unknown pattern: {pattern_name}")
            print(f"Available patterns: {list(pattern_checks.keys())}")
            return None
        
        # Detect this specific pattern
        pattern_info = pattern_checks[pattern_name](threshold)
        
        if pattern_info is None:
            print(f"Pattern '{pattern_name}' not detected in this matrix")
            return None
        
        # Create BlockStructure for this pattern
        blocks = pattern_info.get('blocks')
        if blocks is None:
            blocks = self._create_simple_blocks()
        
        block_struct = BlockStructure(
            pattern_type=pattern_name,
            pattern_info=pattern_info,
            blocks=blocks,
            method='pattern'
        )
        
        # Plot it
        return self.plot_block_structure(block_struct, figsize=figsize, markersize=markersize)
    
    def plot_all_detected_patterns(self, threshold=0.1, figsize=(12, 10), markersize=0.5):
        """
        Plot each detected pattern in separate figures
        Returns list of (pattern_name, figure, axes) tuples
        """
        all_patterns = self.detect_all_patterns(threshold=threshold)
        
        if not all_patterns:
            print("No patterns detected")
            return []
        
        results = []
        
        for pattern_name, pattern_info in all_patterns.items():
            print(f"\nPlotting {pattern_name}...")
            
            # Create BlockStructure for this pattern
            blocks = pattern_info.get('blocks')
            if blocks is None:
                blocks = self._create_simple_blocks()
            
            block_struct = BlockStructure(
                pattern_type=pattern_name,
                pattern_info=pattern_info,
                blocks=blocks,
                method='pattern'
            )
            
            # Plot it
            fig, ax = self.plot_block_structure(block_struct, figsize=figsize, markersize=markersize)
            
            if fig:
                results.append((pattern_name, fig, ax))
        
        return results
    
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
        ax2.set