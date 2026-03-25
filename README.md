# ERFMTDA
We propose a new computational model based on rotative factorization machine to predict tsRNA-disease associations. This is the implementation of ERFMTDA: <br>
Wei Lan, Dong Wang, Wenyi Chen, Xuhua Yan, Qingfeng Chen, Shirui Pan, Yi Pan.  
**ERFMTDA: Predicting tsRNA–disease associations using an enhanced rotative factorization machine.**  
bioRxiv. https://doi.org/10.64898/2026.03.20.713298

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

## Parameter Settings

### Dataset preprocessing
The following parameters are configured in `generate_dataset.py`:
- Number of principal components in the association matrix：32
- Top-k parameter used in the negative sampling module：20

### Model training
The following parameters are specified in `train.py`:
- Feature embedding dimension: 32  
- Hidden units in the attention layers: 32  
- Dropout rate (attention mechanism and amplification network): 0.1  
- Optimizer: Adam  
- Learning rate: 1×10⁻³  
- L2 regularization weight decay: 1×10⁻⁵  
- Batch size: 32  
- Training epochs: 200

