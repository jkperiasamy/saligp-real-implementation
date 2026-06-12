# SALIGP Best Java Implementation

This project contains a non-hardcoded Java implementation inspired by the SALIGP paper.

## Compile

```bash
cd src
javac SALIGPBestImplementation.java
```

## Run

```bash
java SALIGPBestImplementation ../output/saligp_table5_style_result.csv
```

## Output

The program creates a Table-5-style CSV with results for 100, 200, 300, 400, and 500 files:

- precision
- recall
- F1 score
- SALIGP duplicate detection percentage
- execution time

## Important

The paper's exact Table 5 values cannot be reproduced exactly unless the original dataset, source code,
parameters, and random seeds are available. This implementation computes values from generated mixed-content
cloud files with duplicate and near-duplicate samples.
