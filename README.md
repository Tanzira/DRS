# DRS
Repository for breast cancer metastasis analysis.
Global transcriptional rewiring accurately predicts breast cancer metastasis]{Global transcriptional rewiring accurately predicts breast cancer metastasis
# Dataset
## All the intermediary files and datasets can be found here.
10.5281/zenodo.10070758
## NetworkAnalysis.py
- Generate network for a specific lambda using bootstrap
- Read the coefficient files and generates networks
- Generate .gml file for cytoscape to visualize the networks
- Perform statistical analysis

## Training.py
- Generate models for classification using both stratified 10-fold cross-validation and leave-one-study-out cross-validation
- Generate models for NKI dataset classification using 10-fold cross validation
- Generate models with ACES dataset and use NKI as validation dataset.
- Save the models for DRS
- Calculate probability scores for different algorithms and save them

## Figures.py
- Generate figures provided in the paper using saved probability scores for each dataset and configuration

