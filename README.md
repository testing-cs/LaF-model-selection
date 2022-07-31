# LaF: labeling-free comparison testing of deep learning models

## Problem definition

Given N pre-trained deep learning models, the task is to estimate the rank of models regrading their performance on an unlabeled test set.

## Dependency

- python 3.6.10
- keras 2.6.0
- tensorflow 2.5.1
- scipy 1.5.4
- numpy 1.19.5

## Download the dataset

###ID data###
MNIST, CIFAR-10, and Fashion-MNIST are available in Keras.

Amazon and iwildcam are taken from [WILDS](https://github.com/p-lambda/wilds).
 
Java250 and C++1000 are taken from [Project CodeNet](https://github.com/IBM/Project_CodeNet).

###OOD data###

Download the OOD data of MNIST from [Google drive]() or generate it by <pre><code>python gene_mnist.py</code></pre>

Download the OOD data of CIFAR-10 from [Google drive]() or generate it by <pre><code>python gene_cifar10.py</code></pre>

Download the OOD data of Amazon and iwildCam from [WILDS](https://github.com/p-lambda/wilds).

Download the OOD data of Java250 from [Google drive]().

## Download Pre-trained deep learning models

Download all the models from [Google drive]().

You can also train the models for MNIST and CIFAR-10 by running the scripts in **trainModel/mnist** and **trainModel/cifar10**. 

## How to use

To speed the execution and avoid calling the model repeatedly, we first get the model prediction. E.g.:

```
python main_ground.py --dataName mnist
```

To get the results by baseline methods (SDS, Random, CES), run the following code:

```
python main_selection.py --dataName mnist --metric random
```

Besides, to get the final results of CES, you need to run:

```
python main_ces_best.py --dataName mnist
```

To get the results by LaF, run the following code:
```
python main_laf.py --dataName mnist --dataType id
```

To get the evaluation on kendall's tau, spearman's coefficients, jaccard similarity, run the following code:

```
python main_eva.py --dataName mnist 
```

**[Notice] Be careful with the saving directories.**

## Reference
<pre><code>
</code></pre>
