# MotifMol3D
MotifMol3D is a predictive tool for molecular metabolic pathway categories that integrates motif information, graph neural networks, and 3D data to extract local features from small-sample molecules. 

## Requirements
    python 3.7
    torch 0.9.1
    hyperopt 0.2.7
    rdkit
    

## Model usage
### Training data
    Data format: smiles + '\t' + category(represent with corresponding number)
### Code manual
    main.py：main model framwork
    data.py: data preprocessing
    motif_generator.py: split molecules into motifs
    mol_to_molgraph.py: convert molecules to molecular graphs
    hyper.py: model parameters
    GAT_model.py: graph attention network

## Reference
1.Yu Z. and Gao H. Molecular representation learning via heterogeneous motif graph neural networks. In, International Conference on Machine Learning. PMLR; 2022. p. 25581-25594.  
2.Yu J., Wang J., Zhao H., et al. Organic compound synthetic accessibility prediction based on the graph attention mechanism. Journal of Chemical Information and Modeling 2022;62(12):2973-2986.  
