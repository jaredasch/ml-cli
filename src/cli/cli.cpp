#include <iostream>
#include <Eigen/Dense>

#include "types.h"

#include "loaders/loaders.h"
#include "classifiers/binary_logistic_regression.h"
#include "classifiers/logistic_regression.h"

#include "optimizers/batch_gradient_descent.h"
#include "optimizers/stochastic_gradient_descent.h"
#include "optimizers/minibatch_gradient_descent.h"
#include "optimizers/adam_optimizer.h"


int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    for (int i = 0; i < argc - 1; i++) {
        args.push_back(std::string(argv[i+1]));
    }

    if (args[0] == "train") {
        // Usage: ./ml-cli train <data> <labels> <outfile>
        LogisticRegression reg;

        mat x_train = loaders::load_csv(args[1]);
        std::cout << "|" << args[1] << "|" << std::endl;
        std::vector<std::string> y_train = loaders::load_labels(args[2]);

        AdamOptimizer opt = AdamOptimizer(0.1, 200, 1000);

        loaders::add_bias(x_train);
        std::cout << "Started training...";
        reg.fit(x_train, y_train, opt);
        std::cout << "finished" << std::endl;
        reg.export_model(args[3]);
        std::cout << "Model saved to " << args[3] << std::endl;
    } 
    else if (args[0] == "predict") {
        // Usage: ./ml-cli predict <model file> <data> <outfile>

        LogisticRegression reg(args[1]);

        mat data = loaders::load_csv(args[2]);
        loaders::add_bias(data);

        std::vector<std::string> pred = reg.predict(data);
        loaders::export_prediction(pred, args[3]);
    }
    else if (args[0] == "accuracy") {
        // Usage: ./ml-cli acc <predicted> <actual>
        std::vector<std::string> predicted = loaders::load_labels(args[1]);
        std::vector<std::string> actual = loaders::load_labels(args[2]);

        std::cout << "Accuracy: " << LogisticRegression::accuracy(predicted, actual) << std::endl;
    }
    else if (args[0] == "help") {
        std::cout << "Commands: " << std::endl;
        std::cout << "Train a model: ./ml-cli acc <predicted> <actual>" << std::endl;
        std::cout << "Predict with a trained model: ./ml-cli train <data> <labels> <outfile>" << std::endl;
        std::cout << "Compute accuracy: ./ml-cli acc <predicted> <actual>" << std::endl;
    }
}