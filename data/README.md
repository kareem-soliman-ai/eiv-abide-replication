# Data Setup

This repository does not ship the ABIDE-I phenotypic CSV. Download it separately
from the Preprocessed Connectomes Project ABIDE page:

- http://fcon_1000.projects.nitrc.org/indi/abide/

The phenotypic file is commonly distributed with a name similar to
`Phenotypic_V1_0b_preprocessed1.csv`. After downloading it:

1. Place the CSV in this `data/` directory.
2. Rename it to `abide_phenotypic.csv`.

If you keep the original filename, you can still run the pipeline by passing an
explicit path:

```bash
python src/run_analysis.py --input path/to/Phenotypic_V1_0b_preprocessed1.csv
```

The analysis expects the public ABIDE-I phenotypic columns used in the paper,
including `SITE_ID`, `FIQ`, `func_mean_fd`, `AGE_AT_SCAN`, and `qc_rater_1`.

For public reproducibility releases of this repository, keep this source CSV
outside version control and obtain it directly from the ABIDE distribution under
their terms.
