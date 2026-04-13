# The IQ-Motion Confound in Multi-Site Autism fMRI May Be Inflated by Site-Correlated Measurement Uncertainty

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

**Author:** Kareem Soliman

This repository provides a self-contained replication of the ABIDE-I phenotypic analysis reported in the paper. It reimplements only the published equations needed to estimate the pooled OLS slope, the EIV-corrected PCR slope, the leave-site-out cross-validation result, the within-tier slope analysis, and the 8x8 sensitivity grid. 

PCR is introduced as a novel method in this work. This repository contains the implementation of Probability Cloud Regression (PCR) sufficient to reproduce the results reported in the paper. 

## Scope and IP Boundary

- Only the paper-specific ABIDE-I replication is included here.
- The code was written afresh for this repository and does not depend on any other project modules.
  
## Setup

Python 3.9+ is sufficient.

```bash
pip install -r requirements.txt
```

## Data

Download the public ABIDE-I phenotypic CSV separately, then place it in `data/abide_phenotypic.csv`. Detailed instructions are in [data/README.md](data/README.md).

## Usage

```bash
python src/run_analysis.py
```

If the CSV lives elsewhere:

```bash
python src/run_analysis.py --input /path/to/abide_phenotypic.csv
```

## Expected Output

Running the full pipeline will:

- filter the ABIDE-I phenotypic CSV to the paper sample (`n = 935`, 19 sites)
- print the OLS fit, PCR fit, EM convergence trace, Table 2 tier slopes, and
  LOSO cross-validation summary
- save privacy-safe CSV outputs under `results/` (no subject identifiers such as
  `SUB_ID` or `FILE_ID`)
- save four PNG figures under `figures/`
- run a verification block against the reported paper values

At baseline, the expected headline values are:

- OLS slope: about `-0.00125`
- PCR slope: about `-0.00027`
- bias factor: about `4.67x`
- LOSO `R^2`: about `-0.074`

The manuscript reports convergence in 47 iterations for the original analysis run. This rounded-parameter reimplementation converges faster on the public CSV while still matching the reported headline results within tolerance.

## Privacy and Data Distribution

- This repository does not distribute the raw ABIDE-I phenotypic CSV.
- Users must download ABIDE-I from the official source and follow its terms and
  conditions.
- Published outputs in this repo are reduced to reproducibility-focused,
  privacy-safe tables and omit direct subject identifiers.

## Citation

If you use this repository, please cite:

```bibtex
@article{soliman2026iqmotion,
  title={The IQ-Motion Confound in Multi-Site Autism fMRI May Be Inflated
         by Site-Correlated Measurement Uncertainty},
  author={Soliman, Kareem},
  year={2026}
}
```

## Contributing

Reproduction reports and bug fixes are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

**Kareem Soliman** — Independent AI Researcher
- GitHub: [@kareem-soliman-ai](https://github.com/kareem-soliman-ai)
- Email: kareem.soliman@outlook.com.au

## Licence

MIT — see [LICENSE](LICENSE) for details.
