## What is this?
This is a tool to run some ML subroutines from the command line. I'm primarly using this as a learning experience for C++ and how to optimize the implementation of ML algorithms.

I'm also curious about certain ways to parallelize training machine learning algorithms, and hopefully developing them in a systems-heavy language like C++ will allow me to explore this.

## Usage

The command line interface consists of three main commands. 
### Training
The first is `train`, which has the following usage.
```./bin/ml-cli train <data> <labels> <outfile> [<opt-type>]```
`data` should be a filename for  CSV containing only the numerical data to be used, and `labels` should be a filename for a column vector of the correct labels for each training example. `outfile` is the file that the model will be stored in, so that it can be used for training. `opt-type` describes the optimizer used for training, and can be any of the following:
- `adam` - Adam (default)
- `sgd` - Stochastic Gradient Descent
- `minibatch` - Minibatch Gradient Descent
- `batch` - Batch Gradient Descent

Fine tuning of the parameters for each of the optimizers is supported in the underlying implementation, but not yet available in the CLI. 

### Predicting
To predict new data usign an existing model, you should use the `predict` command, which has the following usage.
```./bin/ml-cli predict <model file> <data> <outfile>```
`model file` is the path to the model outputted by the training step, and must be one produced by this program. `data` should be the path to a CSV file of data to predict. The command will produce output in the file at path `outfile`, where it will be represented as a column vector.

### Measuring Accuracy
To compute the accuracy of predicted labels vs. actual labels, use the `accuracy` command, which computes the 0-1 loss between the two provided label vectors. It has the following usage.
```./bin/ml-cli accuracy <predicted> <actual>```
`predicted` and `actual` should both be in column vector format, the same format produces by prediction.

### See Available Commands
If you ever forget a command, just run `./bin/ml-cli help` and it'll descibe all of the available commands.

## Next To-Dos
- Neural networks
- AdaBoost

## Instructions to Run on Ubuntu
- Install the Eigen libary for linear algebra into the `lib` directory
- Run `make` in the root directory, and then call `ml-cli`
