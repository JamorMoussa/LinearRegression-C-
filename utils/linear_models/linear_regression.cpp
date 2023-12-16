#include <iostream>
#include "linear_regression.h"
#include "../data/data_loader.h"


LinearRegression::LinearRegression(double lr, int n_iters){
    this->lr = lr;
    this->n_iters = n_iters;
}

LinearRegression::~LinearRegression(){

}

void LinearRegression::fit(DataFrame X_train, DataFrame y_train){
    this->X_train = X_train;
    this->y_train = y_train;

    X_train = DataFrame::validate_data(X_train);

    //X_train.display();

    auto [n_samples, n_features] = X_train.shape();

    weights = Eigen::MatrixXd::Random(n_features, 1);
    coef_ = Eigen::MatrixXd::Random(n_features - 1, 1);

    //Eigen::Matrix<double, Eigen::Dynamic, 1> y_pred;
    

    for(int i = 0; i< n_iters; i++){
        
        auto y_pred = X_train.data * weights;

        auto dw = (1.0 / n_samples) * X_train.data.transpose() * (y_pred - y_train.data);

        weights -= lr * dw;

        // if(i%500){
        //     std::cout << "iter : " << i << " loss : " << ((y_pred - y_train.data).transpose()*(y_pred - y_train.data)) << std::endl;
        // }

    }

    intercept_(0, 0) = weights(0, 0);

    for (int i = 0 ; i < n_features - 1; ++i){
        coef_(i, 0) = weights(i+1, 0);
    }

    /* w = (Xt * X)^-1 * Xt * y;
      weights = (X_train.data.transpose() * X_train.data).inverse() * X_train.data.transpose() * y_train.data; */

}