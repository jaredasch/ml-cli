#include "classifiers/logistic_regression.h"
#include "types.h"
#include "classifiers/binary_logistic_regression.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <iostream>
#include <fstream>

void LogisticRegression::fit(mat &data, std::vector<std::string> &labels, FirstOrderOptimizer& opt) {
    // Create bidirectional mapping of labels to columns
    this->labels = std::vector<std::string>();
    this->classifiers = std::vector<BinaryLogisticRegression>();

    std::unordered_set<std::string> labels_seen;
    for (int i = 0; i < (int) labels.size(); i++) {
        auto insert_return = labels_seen.insert(labels[i]);
        if (insert_return.second) {
            this->labels.push_back(labels[i]);
        
        }
    }

    // Train each classifier
    for (int i = 0; i < (int) this->labels.size(); i++) {
        // Create one-hot vector for this label

        mat one_hot_labels(data.rows(), 1);
        
        for(int row = 0; row < data.rows(); row++) {
            if (labels[row] == this->labels[i]) {
                one_hot_labels(row, 0) = 1;
            } else {
                one_hot_labels(row, 0) = 0;
            }
        }

        BinaryLogisticRegression bin_classifier(data.cols());

        bin_classifier.fit(data, one_hot_labels, opt);
        this->classifiers.push_back(bin_classifier);
    }
}

std::vector<std::string> LogisticRegression::predict(mat &data) {
    std::vector<mat> class_probs;
    for (int label = 0; label < (int) this->labels.size(); label++) {
        mat label_prob = this->classifiers[label].class_conditional_prob(data);
        class_probs.push_back(label_prob);
    }

    std::vector<std::string> predictions;
    for (int row = 0; row < data.rows(); row++) {
        int max_label_ind = 0;
        for (int label = 0; label < (int) this->labels.size(); label++) {
            if (class_probs[max_label_ind](row, 0) > class_probs[label](row, 0)) {
                max_label_ind = label;
            }
        }
        predictions.push_back(this->labels[max_label_ind]);
    }

    return predictions;
}

double LogisticRegression::accuracy(std::vector<std::string> actual, std::vector<std::string> pred) {
    int incorrect = 0;
    for (int i = 0; i < (int) actual.size(); i++) {
        if (actual[i] != pred[i]) {
            incorrect++;
        }
    }
    return incorrect / ((double) actual.size());
}

// File format, abel name followed by weight vectors on new lines, assumes no spaces in label names, ex: 
// positive 1 1 0
// negative 2 2 1

void LogisticRegression::export_model(std::string path) {
    std::ofstream out_file;
    out_file.open(path);
    for (int label = 0; label < (int) this->labels.size(); label++) {
        out_file << this->labels[label] << " ";
        for (int weight = 0; weight < this->classifiers[label].get_param("w").rows(); weight++) {
            out_file << this->classifiers[label].get_param("w")(weight, 0) << " ";
        }
        out_file << std::endl;
    }
}

LogisticRegression::LogisticRegression(std::string path) {
    std::ifstream in_file;
    in_file.open(path);

    char separator = ' ';
    int dim = -1;

    if (in_file.good()) {
        std::string line_buf;
        while (getline(in_file, line_buf)) {
            std::vector<double> weight_vec;

            std::stringstream line_stream(line_buf);

            std::string label;
            getline(line_stream, label, separator);

            // Read until seperator
            std::string val;
            while (getline(line_stream, val, separator)) {
                weight_vec.push_back(stod(val));
            }

            // Check that each row has the same number of cols
            if (dim == -1) {
                dim = weight_vec.size();
            } else if ((int) weight_vec.size() != dim) {
                std::cout << "Something's wrong\n";
            }

            // Convert vector to matrix
            mat weight_mat(dim, 1);
            for(int row = 0; row < dim; row++) {
                weight_mat(row, 0) = weight_vec[row];
            }

            BinaryLogisticRegression binary_classifier(dim);
            binary_classifier.update_param("w", weight_mat);

            this->labels.push_back(label);
            this->classifiers.push_back(binary_classifier);
        }
    }
}
