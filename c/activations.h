/*
Activations
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double d_sigmoid(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

double softmax(double x, double total)
{
    return exp(x) / total;
}