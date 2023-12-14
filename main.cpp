#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils/data/data_loader.h"
#include "utils/linear_models/linear_regression.h"

using namespace Eigen;

int main(int argc, char* argv[]){

    DataFrame df = DataFrame::read_csv("./linear_data.csv", ',');
    //df.display();

    // auto [rows, cols] = df.shape();

    // std::cout << rows << " : " << cols << std::endl;

    auto [X_train, y_train] = df.split(0);

    // X_train.display();

    // std::cout << "\n" << std::endl;

    // y_train.display();

    LinearRegression lg(0.01, 1000);

    lg.fit(X_train, y_train);

    std::cout << "w : " << lg.weights << std::endl;
    std::cout << "b : " << lg.bias << std::endl;

}