Source code for the paper _An Integrated Approach to Knowledge and Prediction Modeling of Breast Cancer Metastasis Using Gene Regulatory Networks_
## Datasets and Results
All the intermediary files and datasets can be found at https://drive.google.com/drive/folders/1QmVE96fgWuBzlYo06aggyiJY0JFUARvl
## Files
- `drs.py`: core module for learning LASSO regression models of gene expression, saving and loading coefficients, and using the dysregulation score to classify new samples.
- `training.py`: load data and train DRS models.
- `training_external_grns.py`: infer GRNs using [GReNaDIne](https://grenadine.readthedocs.io/en/latest/) for feature preselection prior to training DRS models.
- `network_analysis.py`: downstream analysis of the GRNs obtained from training.
- `figures.py`: generate manuscript plots.
- `utils.py`: utility functions
