Attention Bidirectional Grident Recurrent Unit(Att-BiGRU) Memory Networks
===============================================

An implementation of the Att-BiGRU architectures

## Requirements


- Python >= 3.5
- pytorch >= 0.2.0
- matplotlib >= 2.2.0
- pandas >= 0.12.0
- tqdm >= 4.23.4
The pytorch dependencies can be installed using pip . For example:

```
pip install matplotlib
```

## Usage

This downloads the following data:

  - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)

### Semantic Relatedness

The goal of this task is to predict similarity ratings for pairs of sentences. We train and evaluate our models on the [Sentences Involving Compositional Knowledge (SICK)](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) dataset.
