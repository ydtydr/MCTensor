# Hyperbolic Embedding using MCTensor
This module contains official code and models to reproduce Section *3.2 Hyperbolic Embedding* in our paper. We implemented our models under the same frame of [Poincaré Embeddings for Learning Hierarchical Representations](https://github.com/facebookresearch/poincare-embeddings). The Poincaré Halfspace model is implemented based on [Representing Hyperbolic Space Accurately using Multi-Component Floats](https://proceedings.neurips.cc/paper/2021/hash/832353270aacb6e3322f493a66aaf5b9-Abstract.html), in both PyTorch Tensor and MCTensor. We conduct hyperbolic embedding experiments on the [WordNet](https://wordnet.princeton.edu/) Mammals dataset with 1181 nodes and 6541 edges, and results for running the two versions of the model for 5 seeds can be found in *Table 2* in the paper. 

## Install
### Requirements
Same as installing *MCTensor*, and:
- Scikit-Learn
- NLTK (to generate the WordNet data)

### Procedures
1. cd to this directory: `cd applications/poincare_embedding`
2. run following commands:
```python
conda env create -f environment.yml
conda activate poincare
python setup.py build_ext --inplace 
cd ../..
python build.py install
```
3. Generate the transitive closure of the WordNet Mammals dataset:
```python
cd applications/poincare_embedding/wordnet
python transitive_closure.py
```

## Training hyperbolic embedding
To train and evaluate (MC)Halfspace embeddings, run notebooks `Halfspace.ipynb` and `MCHalfspace.ipynb`. Included in the repo is the Pytorch Tensor and MCTensor version of the Poincaré Halfspace embedding for the WordNet Mammals dataset. Both model weight are initalized with float64. Particularly, we set MCTensor with *nc=2* components.

### Hyperparameters
To modify the hyperparameters or the datatype for each model, edit the cell below _Hyperparameter_ section in each notebook. The *number of componenets* or *nc* for the MCTensor model can also be easily edited with the variable name `opt_nc`. Note that with the same seed, `MCHalfspace` and `Halfspace` embeddings' weights are initalized equally: the `MCHalfspace` model's first componenet (`fc`) is equal to the `Halfspace` model's weight, with all other componenets equal to zeros. Therefore, fair comparison between models can be achieved as long as both models have the same seeds.