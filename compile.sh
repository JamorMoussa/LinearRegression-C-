#!/bin/bash

echo "compiling started ...."

g++ -c main.cpp -o main.o

g++ -c utils/linear_models/linear_regression.cpp -o utils/linear_models/linear_regression.o
g++ -c utils/linear_models/multiple.cpp -o utils/linear_models/multiple.o

g++ -c utils/data/data_loader.cpp -o utils/data/data_loader.o

g++ main.o utils/data/data_loader.o utils/linear_models/linear_regression.o utils/linear_models/multiple.o -o main 


echo "finished"