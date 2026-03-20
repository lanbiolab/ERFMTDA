# ERFMTDA
We propose a new computational model based on rotative factorization machine to predict tsRNA-disease associations. This is the implementation of ERFMTDA: <br>
**Wei Lan, Dong Wang, Wenyi Chen, Xuhua Yan, Qingfeng Chen, Shirui Pan, Yi Pan** <br>
*ERFMTDA: Predicting tsRNA–disease associations using an enhanced rotative factorization machine*

## Environment Requirement
- Python 3.12.7
- PyTorch 2.5.1
- NumPy 1.26.4
- pandas 2.2.2
- matplotlib 3.9.2

## Dataset
The dataset used in this study was manually curated from published literature. We searched the relevant studies and collected experimentally validated 
tsRNA–disease associations. The curated dataset is provided as `tsRNA-disease.xlsx` and is included in the `ERFMTDA`, `case study`, and `denovo` folders for different experimental settings.

## Usage
1. Data preprocessing  
   First run `generate_dataset.py` to preprocess the tsRNA–disease association data.
2. Model training  
   Then run `train.py` to train the model and output the prediction performance metrics.

