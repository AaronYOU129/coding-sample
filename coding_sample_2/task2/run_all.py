# Purpose:    Run the Task 2 pipeline in research order so exploratory evidence
#             informs interpretation and diagnostics use current estimates.
# Inputs:     ../data/mixture_data.csv through `eda.main` and `estimate.main`.
# Outputs:    All files in task2/work, task2/output, and task2/figures through
#             the stage-specific modules.
# Key Steps:  Exploratory analysis -> estimate K=2 -> estimate K=3 and compare
#             AIC/BIC -> run goodness-of-fit diagnostics.
# How to Run: `python run_all.py` from the task2/ directory.

import diagnostics
import eda
import estimate
import estimate_k3


def main() -> None:
    """Run EDA, K=2, K=3 comparison, and goodness-of-fit diagnostics in order.

    The explicit sequence documents the research workflow and prevents stale outputs.
    """
    eda.main()
    estimate.main()
    estimate_k3.main()
    diagnostics.main()


if __name__ == "__main__":  # pragma: no cover
    main()
