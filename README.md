## Overview

This repository contains the source code and supplementary material for the paper **"Process-Aware Temporal Feature Selection for Predictive Process Monitoring"**. The paper investigates how temporal characteristics of business processes—especially seasonal patterns—can be systematically incorporated into feature engineering to improve remaining-time prediction in Predictive Process Monitoring (PPM). We introduce a rule-based guideline for selecting temporal features based on process properties and demonstrate that it outperforms established, uniform feature sets in both synthetic and real-world scenarios.

## Repository Structure

├─ src/
│ └─ experiment/
│ └─ experiment_configs/
├─ imgs/
└─ supplementary/


- **`src`** – Source code required to reproduce the analyses presented in the paper.  
  - The modules in `src/experiment` enable running **hyperparameter optimization**, **recursive feature elimination (RFE)**, and **single validation experiments**.  
  - Experiment execution is configured via YAML files in `src/experiment/experiment_configs`.

- **`imgs`** – Images and figures included in the paper.

- **`supplementary`** – Additional analyses and results not discussed in the publication.

---

For questions or issues, please reach out to the authors.