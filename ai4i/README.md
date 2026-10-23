# AI4I 2020 Predictive Maintenance Dataset

`ai4i2020.csv` is the unmodified dataset file from the UCI Machine Learning
Repository:

> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
> Applications", Third International Conference on Artificial Intelligence for
> Industries (AI4I 2020), 2020.
> https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Keep this
attribution if you redistribute it.

- 10,000 rows, one per machining process, with no timestamps.
- 339 machine failures, by mode: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.
- SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`

The data is **synthetic**. It was generated to resemble a real milling machine,
and its failure labels come from the rules described in
`knowledge/failure_modes.md`. That is why those rules reproduce the HDF, PWF and
OSF labels exactly (`tests/test_failure_rules.py`).
