# Purpose:    Reproduce Task 3 from raw data in a fixed order so no output depends
#             on a manually executed or stale intermediate file.
# Inputs:     Raw CSV files in data/ and the four task3 pipeline stages.
# Outputs:    All files under task3/work/, output/, and figures/.
# Key Steps:  Prepare analysis data -> run exploratory analysis -> estimate
#             models -> create the coefficient figure.
# How to Run: `python run_all.py` from task3/.

import diagnostics
import eda
import estimate
import prepare_data


# Run every stage in research order, matching the Task 1 and Task 2 entry points.
def main() -> None:
    prepare_data.main()
    eda.main()
    estimate.main()
    diagnostics.main()


if __name__ == "__main__":  # pragma: no cover
    main()
