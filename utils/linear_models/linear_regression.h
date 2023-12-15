#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H
#include "../data/data_loader.h"

class LinearRegression{

    private:
        DataFrame X_train;
        DataFrame y_train;
        double lr = 0.01;
        int n_iters = 1000;

    public:
        Eigen::Matrix<double, Eigen::Dynamic, 1> weights;

    public:
        LinearRegression(double lr, int n_iters);
        ~LinearRegression();

        void fit(DataFrame X_train, DataFrame y_train);
};

#endif