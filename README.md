# ERFMTDA
We propose a new computational model based on rotative factorization machine to predict tsRNA-disease associations. This is the implementation of ERFMTDA: <br>
**Wei Lan, Dong Wang, Wenyi Chen, Xuhua Yan, Qingfeng Chen, Shirui Pan, Yi Pan** <br>
*ERFMTDA: Predicting tsRNA–disease associations using an enhanced rotative factorization machine*

## Environment Requirement
- Python 3.8
- PyTorch 1.4.0
- NumPy 1.21.6
- pandas 1.4.2
- matplotlib 3.5.2

## Dataset
The dataset used in this study was manually curated from published literature. 
We searched the relevant studies and collected experimentally validated 
tsRNA–disease associations. The curated dataset is provided in the file 
`tsRNA-disease.xlsx` in this repository.

## Usage
- Data preprocessing: generate_dataset.py
- Train the model: train.py

