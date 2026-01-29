import numpy as np

from .defs import INPUTBLOCKS


class StructureScore:
    """
    a base class for the structure scores
    """

    def __init__(self, equ_blocks, var_blocks):
        self.equ_blocks = equ_blocks
        self.var_blocks = var_blocks

        self.score = 0
        self.scorecomputed = False

    def get_score(self):
        """
        returns the score for the structure. If the score has not be computed,
        then it will be computed before returned

        Returns
        -------
        float
            the score for the structure
        """
        if not self.scorecomputed:
            self.score = self._compute_score()
            self.scorecomputed = True

        return self.score

    def _compute_score(self):
        """
        public function to compute the for the structure

        Returns
        -------
        float
            a score for the structure
        """
        pass


class WhiteScore(StructureScore):
    """
    computes a score of the amount of white space in the matrix area. The white
    space is that which is not the border or a block.
    """

    def __init__(self, equ_blocks, var_blocks):
        super().__init__(equ_blocks, var_blocks)

    def _compute_score(self):
        """
        public function to compute the for the structure

        Returns
        -------
        float
            a score for the structure
        """
        # getting the sort indices for the equations and variables
        equ_sorted = self.equ_blocks.argsort(kind="mergesort")
        var_sorted = self.var_blocks.argsort(kind="mergesort")

        # getting new array's for the equation and variable blocks that
        # correspond to the sorted lists
        equ_stage = self.equ_blocks[equ_sorted]
        var_stage = self.var_blocks[var_sorted]

        # we need to compute matrix area and the block area
        matrix_area = len(equ_stage) * len(var_stage)

        # if there are no variables or no constraints, then the score is 1
        if matrix_area == 0:
            return 1

        block_area = 0

        blocks = np.union1d(np.unique(equ_stage), np.unique(var_stage))

        for block in blocks:
            if block == 1:
                # the linking variables
                block_area += np.count_nonzero(
                    var_stage == 1
                ) * np.count_nonzero(equ_stage < blocks.max())
            elif block == blocks.max():
                # the linking constraints
                block_area += np.count_nonzero(var_stage) * np.count_nonzero(
                    equ_stage == blocks.max()
                )
            else:
                # the blocks
                block_area += np.count_nonzero(
                    var_stage == block
                ) * np.count_nonzero(equ_stage == block)

        return 1 - float(block_area) / float(matrix_area)


class PipsScore(StructureScore):
    """
    a score that finds structures that are suitable for PIPS. Specifically,
    this score computes the white space for the structure and penalises linking
    constraints/variables that link 3 or more blocks.
    """

    def __init__(self, model_dump, equ_blocks, var_blocks):
        super().__init__(equ_blocks, var_blocks)
        self.model_dump = model_dump
        self.whitescorer = WhiteScore(equ_blocks, var_blocks)

        self.num_blocks = int(
            max(np.max(self.equ_blocks), np.max(self.var_blocks))
        )

    def _compute_score(self):
        """
        public function to compute the for the structure

        Returns
        -------
        float
            a score for the structure
        """
        # getting the white score for the structure
        whitescore = self.whitescorer.get_score()

        # computing the linking equations score
        linking_equ_score, linking_equ_frac = self._linking_score(
            INPUTBLOCKS.EQUATIONS
        )

        # computing the linking equations score
        linking_var_score, linking_var_frac = self._linking_score(
            INPUTBLOCKS.VARIABLES
        )

        print(
            "PIPS Score:",
            whitescore,
            linking_equ_score,
            linking_equ_frac,
            linking_var_score,
            linking_var_frac,
        )

        # returning a linear regression based combination of the white score
        # and the linking scores
        return (
            0.85 * whitescore
            + 2 * linking_equ_score
            + -90 * linking_equ_frac
            + -3 * linking_var_score
            + -90 * linking_var_frac
        )

    def _linking_score(self, linking_block_type):
        """
        computes the score for the linking blocks. This is proportional to the
        percentage of the linking region that has variables/constraints that
        link 3 or more blocks

        Returns
        -------
        float
            a score for the linking region
        """
        linking_labels = (
            self.equ_blocks
            if linking_block_type == INPUTBLOCKS.EQUATIONS
            else self.var_blocks
        )
        target_labels = (
            self.var_blocks
            if linking_block_type == INPUTBLOCKS.EQUATIONS
            else self.equ_blocks
        )

        linking_type = (
            "i" if linking_block_type == INPUTBLOCKS.EQUATIONS else "j"
        )
        target_type = (
            "j" if linking_block_type == INPUTBLOCKS.EQUATIONS else "i"
        )

        label = (
            self.num_blocks
            if linking_block_type == INPUTBLOCKS.EQUATIONS
            else 1
        )

        elem_to_block = {
            i: j
            for i, j in zip(
                self.model_dump[target_type].records["uni"], target_labels
            )
        }

        # Get A matrix
        linking_grouping = (
            self.model_dump["A"]
            .records.groupby(linking_type, observed=True)[target_type]
            .apply(list)
            .values
        )

        linking_target_labels = [
            [elem_to_block.get(e) for e in elems]
            if linking_labels[i] == label
            else None
            for i, elems in enumerate(linking_grouping)
        ]

        assert len(linking_labels) == len(linking_grouping)

        def num_linking(target_blocks):
            unique_blocks = np.unique(target_blocks)
            if len(unique_blocks) == 1 and unique_blocks[0] == 1:
                return 10
            else:
                return len(unique_blocks)

        # Get equation annotation
        linking_count = np.array(
            [
                num_linking(target_blocks) if target_blocks is not None else 0
                for i, target_blocks in enumerate(linking_target_labels)
            ]
        ).astype(float)

        total_linking = np.count_nonzero(linking_count)
        # if there are no linking variables/constraints, then there the score
        # is 1
        if total_linking == 0:
            return (1, 0)

        three_plus_linking = np.count_nonzero(linking_count >= 3)
        # import pdb; pdb.set_trace()

        return (
            1 - float(three_plus_linking) / float(total_linking),
            float(total_linking) / float(len(linking_labels)),
        )
