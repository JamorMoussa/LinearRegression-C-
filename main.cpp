#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils/data/data_loader.h"

using namespace Eigen;

int main(int argc, char* argv[]){

    DataFrame df = DataFrame::read_csv("./data.csv", ',');
    //df.display();

    auto [X_train, y_train] = df.split(0);

    X_train.display();

    std::cout << "\n" << std::endl;

    y_train.display();

}