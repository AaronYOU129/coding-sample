# Purpose:    Run the full Task 1 pipeline in order. Exists because the
#             modules were renamed to importable names (eda, prepare_data,
#             estimate, diagnostics) for testing, so execution order is
#             documented here instead of in numeric filename prefixes.
# Inputs:     ../data/orders_sample.csv through eda and prepare_data.
# Outputs:    Everything in work/, output/, figures/ through the four modules.
# Key Steps:  Exploratory analysis -> cleaning -> estimation -> diagnostics;
#             later stages communicate through files in work/ and output/.
# How to Run: python run_all.py   (from the task1/ directory)

import diagnostics
import eda
import estimate
import prepare_data


def main() -> None:
    eda.main()
    prepare_data.main()
    estimate.main()
    diagnostics.main()


if __name__ == "__main__":  # pragma: no cover
    main()
