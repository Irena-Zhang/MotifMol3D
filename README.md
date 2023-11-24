# MotifMol3D
MotifMol3D is a predictive tool for molecular metabolic pathway categories that integrates motif information, graph neural networks, and 3D data to extract local features from small-sample molecules. 

## Requirements
    Pytorch1.9

## Model usage
### Training data
    Data format: smiles + '\t' + category(represent with corresponding number)
### Code manual
    gasa_motif_trans_structure_descri_3d_feature_aro_atom_metric.py：main model framwork
    data.py: data preprocessing
    motif_generator.py: split molecules into motifs
    gasa_utils_aro.py: convert molecules to molecular graphs
    hyper.py: model parameters
    aro_model_metric.py: graph attention network
