# Scalable Machine Learning Pipeline
[![Build Status](https://travis-ci.com/IBM/smlp.svg?branch=master)](https://travis-ci.com/IBM/smlp)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Scope

Efficient, scalable Machine Learning pipeline, that enables training and inference of large datasets that do not fit in memory by scaling up using fast storage.

Builds a ML pipeline on top of existing ML libraries ([IBM Snap ML](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/), [scikit-learn](https://github.com/scikit-learn/scikit-learn)), and using the AWS [ML-IO](https://github.com/awslabs/ml-io) library.

## Usage


## Setup conda environment
```
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --add channels conda-forge
conda config --add channels mlio
conda create --yes -n smlp-environment python=3.7
conda activate smlp-environment
```

## Install dependencies
```
conda install --file requirements.txt --yes
```

## Install smlp module locally

```
python setup.py install
```

## Run a test

```
python test/MLPipelineTester.py --ml_lib snap
```

## Full pipeline test example

### Epsilon dataset from the [PASCAL Large Scale Learning Challenge](http://www.k4all.org/project/large-scale-learning-challenge/).
```
for ch in 50000 100000 200000; do echo "chunk="$ch; python examples/smlp-demo.py --dataset_path /path_to_dataset/epsilon.train.csv --dataset_test_path /path_to_dataset/epsilon.test.csv --chunk_size $ch --ml_lib snap --ml_obj logloss --ml_model_options objective=logloss,num_round=1,min_max_depth=4,max_max_depth=4,n_threads=40,random_state=42; echo; done
```

## Notes

Currenlty we support:
- ML models: Snap Booster, sklearn Decision Trees
- Input data format: csv

### Dependencies:

- Python (>= 3.7)
- scikit-learn
- numpy
- pai4sk
- mlio-py
- psutil

## License

This project is licensed under the Apached 2.0 License.
If you would like to see the detailed LICENSE click [here](LICENSE).

## Contributing

Please see [CONTRIBUTING](CONTRIBUTING.md) for details.
Note that this repository has been configured with the [DCO bot](https://github.com/probot/dco).
