import random
from typing import List


def generate_placeholder_matrix(rows: int, cols: int) -> List[List[float]]:
    matrix = []
    for i in range(min(rows, 20)):
        row = [
            random.choice([0, 0, 0, 0,
                           round(random.uniform(-10, 10), 2)])
            for j in range(min(cols, 15))
        ]
        matrix.append(row)
    return matrix
