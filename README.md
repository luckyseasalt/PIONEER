# An Unsupervised Learning Framework Combined with Heuristics (PIONEER) for the Maximum Minimal Cut Problem (MMCP)
The official implementation of the paper “An Unsupervised Learning Framework Combined with Heuristics for the Maximum Minimal Cut Problem”.


## Datasets
We mainly use real-world datasets and synthetic datasets in this study.

**Real-world Datasets**

The real-world datasets mainly contain the following three datasets.

* [ENZYMES](https://paperswithcode.com/dataset/enzymes)

* [IMDB](http://www.graphlearning.io/)

* [REDDIT](https://www.reddit.com/)

**Synthetic Datasets**

The synthetic datasets utilize the images in the [Mnist](http://yann.lecun.com/exdb/mnist/) dataset.

- To generate the synthetic datasets:

Fill in the path that you would like to put the data in './build_dataset/configs/config'.

Enter './build_dataset', run

```Python
python build_data.py
```

# Usage
- To evaluate the PIONEER on the real-world datasets, run

```Python
python main.py
```

- To further assess the PIONEER on the synthetic datasets, enter './train_atheta' and use the following command to train:

```Python
python train.py
```

Then, run the following command to test:

```Python
python test.py
```

The proposed heuristic tree transformation can be assessed by running:

```Python
python test_without_us.py
```

- The baselines contain a brute force search algorithm, random tree search algorithm, and genetic algorithm that can be respectively evaluated by applying:

```Python
python brute_force.py
```

```Python
python random_tree.py
```

```Python
python test_ga.py
```

# Environment Requirements
The following packages are required to install to implement our code:
```shell
conda create -n pioneer python=3.7.11
conda activate pioneer
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111   # this may take a while...
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111
pip install torch_geometric==1.7.2

Optional but recommend:
pip install matplotlib
pip install pyyaml
pip install tensorboardx
```
