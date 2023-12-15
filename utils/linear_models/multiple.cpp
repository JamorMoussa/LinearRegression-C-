#include "multiple.h"
#include "../data/data_loader.h"
#include "linear_regression.h"

LinearRegressionMultiple::~LinearRegressionMultiple(){}


void LinearRegressionMultiple::fit(DataFrame X_train, DataFrame y_train){

    auto [n_samples , n_features] = X_train.shape();

    auto [_ , y_dim] = y_train.shape();

    //weights = Eigen::MatrixXd::Random(n_features, y_dim);

    LinearRegression lg(lr, n_iters);

    Eigen::Matrix<double, Eigen::Dynamic, 1> y_trainVect;

    for(int j = 0; j < y_dim ; j++){

        y_trainVect = y_train.data.col(j);

        lg.fit(X_train, DataFrame(y_trainVect));

        std::cout << lg.weights << std::endl;
    }
}