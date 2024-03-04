# MotifMol3D
MotifMol3D is a predictive tool for molecular metabolic pathway categories that integrates motif information, graph neural networks, and 3D data to extract local features from small-sample molecules. <br/>

## Requirements
* python 3.6+ <br/>
* DGL 0.7.0+ [https://www.dgl.ai/pages/start.html]<br/>
* PyTorch 1.5.0+[https://pytorch.org/get-started/locally/]<br/>
* dgllife 0.2.6+ [https://github.com/awslabs/dgl-lifesci]<br/>
* RDKit (recommended version 2018.03.1+) [https://github.com/rdkit/rdkit]

## Model usage
### Training data
 * Data format: smiles + '\t' + category(represent with corresponding number)
 * Dataset for training, validation and test the model <br/>
 * three external test sets:TS1, TS2 and TS3 <br/>
 
### Code manual
 * data.py: import and process the data <br/>
 * model.py: define MotifMol3D models <br/>
 * gasa_utils_aro.py: converts SMILES into graph with features <br/>
 * hyper.py: code for hyper-parameters optimization <br/>
 * gasa.json: best combination of hyper-parameters for MotifMol3D <br/>

### Test
* main.py: code for predicting the results for given molecules <br/>
* explain.ipynb: atom weights visualization for given compound <br/>

## Reference
1.Yu Z. and Gao H. Molecular representation learning via heterogeneous motif graph neural networks. In, International Conference on Machine Learning. PMLR; 2022. p. 25581-25594.  
2.Yu J., Wang J., Zhao H., et al. Organic compound synthetic accessibility prediction based on the graph attention mechanism. Journal of Chemical Information and Modeling 2022;62(12):2973-2986.  
