# An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality

### [Silviu Pitis](https://silviupitis.com)\*, [Harris Chan](https://takonan.github.io/)\*, Kiarash Jamali, Jimmy ba

This repo contains Tensorflow v1 and Pytorch implementations of **Deep Norms**, **Wide Norms**, and **Neural Metrics**, from our ICLR 2020 paper, which may be found at:

- [Openreview](https://openreview.net/forum?id=HJeiDpVFPr)
- [Arxiv](https://arxiv.org/abs/2002.05825)


### Bibtex


```
@inproceedings{pitis2020inductive,
  title={An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality},
  author={Pitis, Silviu and Chan, Harris and Jamali, Kiarash and Ba, Jimmy},
  booktitle={Proceedings of the Eighth International Conference on Learning Representations},
  year={2020}
}
```

### Installing dependencies
To create a conda/virtualenv and install dependencies for our code, assuming that `${DEEPNORM_ROOT}` the directory that you cloned our repository into:
```
conda create -n deepnorm
conda activate deepnorm
cd ${DEEPNORM_ROOT}
pip install -r requirements.txt
```

### File Descriptions

The main implementation files are:
- `metrics_tf1.py`: defines the model components in Tensorflow v1 code
- `metrics_pytorch.py`: reimplementations of the model components in Pytorch (untested)
- `custom_metric_loss_ops.py`: drop in replacement for Tensorflow v1's `contrib.metric_loss_ops` that computes pairwise distances using our models

### Pytorch examples

To run Pytorch examples, use `pytorch_examples.ipynb`. 

### Additional files for graph experiments

Some additional files to reproduce some of our experiments (**Please note:** this is TFv1 research code, there are no guarantees that this will "just work".):

- `Make3dAndTaxi.ipynb`: creates 3d and Taxi graphs in form of .edgelist
- `PrepXYDPickles.ipynb`: creates the dataset (but not embeddings)
- `data.py`: makes landmark embeddings given a .edgelist, and loads datasets used in experiments
- `experiment.py`: runs experiment given an embedding pickle and a dataset pickle

### How to run graph experiments

1. Use `Make3dAndTaxi.ipynb` to make `.edgelist` 
2. Use `PrepXYDPickles.ipynb` given `.edgelist`
3. Use `make_lm_embeddings` from `data.py` given `.edgelist`
4. Use `experiment.py` to run experiment using pickes created in previous two steps

### How to run 2D Norms experiment

1. Run `python experiment_2d_norm.py`. This trains several architectures on different 2D norms and saves the results in `./2D_metrics` folder.
2. Launch the notebook `2D Norm Result Plotting.ipynb` and run the cells to update the plots for the results. 