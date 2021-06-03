./bin/ml-cli train data/ex1/X_test.txt data/ex1/y_test.txt model batch
./bin/ml-cli predict model data/ex1/X_train.txt test_out
./bin/ml-cli accuracy data/ex1/y_train.txt test_out