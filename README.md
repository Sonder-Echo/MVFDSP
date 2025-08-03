# MVFDSP
MVFDSP is a deep learning model we proposed to predict the frequency of drug side effects.

# Requirements
* networkx==2.8.8
* numpy==1.22.4
* pandas==1.5.1
* rdkit==2022.9.4
* rdkit-pypi==2022.9.4
* scikit_learn==1.2.1
* scipy==1.10.1
* torch==1.12.1+cu113
* torch-geometric==1.7.2
* torch-cluster==1.6.0
* torch-scatter==2.0.9
* torch-sparse==0.6.15
  
# Files:
1.data

This folder contains original side effects and drugs data.

* **frequency_data.txt:**
  The standardised drug side effect frequency classes used in our study.

* **drug_SMILES_750.csv:**
  The SMILES representations of 750 drugs.

 * **raw_frequency_750.mat:**
   The original matrix of drug-adverse effect frequencies.

* **side_effect_label_750.mat:**
  The encoded features of side effects.

* **mask_mat_750.mat:**
  The mask matrix for ten-fold cross-validation in a warm-start scenario.


2.data_processed

This folder contains preprocessed data.
   
* **processed:**
  This folder contains side effect and drug data used for 10-fold cross-validation.

* **drug_feature_1d.pt & drug_feature_2d.pt:**
  Preprocessed 1D and 2D features of 750 drugs

* **drug_feature_1d_new9.pt & drug_feature_2d_new9.pt:**
  Preprocessed 1D and 2D features of 9 drugs

  
# Code 
model.py: It defines the model.

vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.

utils.py: It defines some other functions.

train.py: Training for the frequency of drugs and side effects.


# Run
```bash
python train.py --tenfold --save_model --epoch 12000 --lr 0.00005
```
