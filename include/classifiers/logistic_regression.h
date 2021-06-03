#include "types.h"
#include "optimizers/first_order_optimizer.h"
#include "classifiers/binary_logistic_regression.h"

#include <vector>
#include <unordered_map>

class LogisticRegression  {
    public:
        LogisticRegression() {};

        LogisticRegression(std::string path);

        void fit(mat &data, std::vector<std::string> &labels, FirstOrderOptimizer* opt);

        std::vector<std::string> predict(mat &data);

        static double accuracy(std::vector<std::string> actual, std::vector<std::string> pred);

        void export_model(std::string path);

    private:
        // Map from column number to string 
        std::vector<std::string> labels;

        // Individual classifiers
        std::vector<BinaryLogisticRegression> classifiers;
};