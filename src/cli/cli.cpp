#include <iostream>
#include <armadillo>

#include "loaders/loaders.h"

using namespace std;
using namespace arma;
using namespace loaders;

int main() {
    string path;
    while(1) {
        cout << "Enter a file path to open: ";
        cin >> path;
        load_csv(path);
    }
}