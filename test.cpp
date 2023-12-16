#include <iostream>
#include <eigen3/Eigen/Dense>
#include <omp.h>

int main() {
    int n_features = 10; // Replace with your desired size
    int y_dim = 3; // Replace with the number of matrices you want to stack

    // Create the first matrix of size (n_features, y_dim)
    Eigen::MatrixXd firstMatrix;
    
    firstMatrix = Eigen::MatrixXd::Random(n_features, 1);

    Eigen::MatrixXd finalMatrix = Eigen::MatrixXd::Random(n_features, y_dim);

    #pragma omp parallel for 
    for (int i = 0; i < y_dim; ++i) {
        std::cout << i << std::endl;
        finalMatrix.col(i) = firstMatrix.col(0);
    }


    // Print the result
    std::cout << "First Matrix:\n" << finalMatrix << std::endl;

    return 0;
}
