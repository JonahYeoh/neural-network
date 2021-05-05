/*
Model
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "activations.h"

typedef struct Neuron
{
    int idx, n_weights;
    double *weights;
    double bias;
    double delta;
    double *inputs;
    double output;
    double activated_output;
} NEURON;

typedef struct Layer
{
    int idx;
    int n_neurons;
    NEURON **neurons;

} LAYER;

typedef struct Network
{
    int n_features, n_classes;
    int n_layers;
    LAYER **layers;
    double learning_rate;
} NETWORK;

NEURON * init_neuron(const int i, const int in)
{
    NEURON *neuron = (NEURON*)malloc(sizeof(NEURON));
    neuron->idx = i;
    neuron->n_weights = in;
    neuron->weights = (double*)malloc(sizeof(double)*in);
    neuron->bias = ((rand() % 1000000) - 500000) * 0.0000001;
    neuron->delta = 0.0;
    neuron->inputs = (double*)malloc(sizeof(double)*in);
    neuron->output = 0.0;
    neuron->activated_output = 0.0;
    int j;
    for (j=0; j<in; j++)
    {
        neuron->weights[j] = ((rand() % 1000000) - 500000) * 0.0000001;
    }
    return neuron;
}

void release_neuron(NEURON * neuron)
{
    free(neuron->weights);
    free(neuron->inputs);
    free(neuron);
}

void print_neuron_weights(NEURON * neuron)
{
    int j;
    printf("(");
    for (j=0; j<neuron->n_weights; j++)
        printf("%.6lf,", neuron->weights[j]);
    printf(")");
}

double get_neuron_output(NEURON * neuron, const double *inputs)
{
    neuron->output = 0;
    int j;
    for (j=0; j<neuron->n_weights; j++)
    {
        neuron->inputs[j] = inputs[j];
        neuron->output += neuron->weights[j] * inputs[j];
    }
    neuron->output += neuron->bias;
    return neuron->output;
}

LAYER * init_layer(const int i, const int n_neurons, const int n_features)
{
    LAYER * layer = (LAYER*)malloc(sizeof(LAYER));
    layer->idx = i;
    layer->n_neurons = n_neurons;
    layer->neurons = (NEURON**)malloc(sizeof(NEURON*)*n_neurons);
    int j;
    for (j=0; j<n_neurons; j++)
    {
        layer->neurons[j] = init_neuron(j, n_features);
    }
    return layer;
}

void release_layer(LAYER * layer)
{
    int j;
    for (j=0; j<layer->n_neurons; j++)
    {
        release_neuron(layer->neurons[j]);
    }
    free(layer);
}

void print_layer_weights(LAYER *layer)
{
    int j;
    printf("\nlayer: %d\n", layer->idx);
    printf("[");
    for (j=0; j<layer->n_neurons; j++)
        print_neuron_weights(layer->neurons[j]);
    printf("]");
}

double * get_layer_output(LAYER * layer, const double *inputs)
{
    double * output = (double*)malloc(sizeof(double)*layer->n_neurons);
    double tmp, activated;
    int j;
    for (j=0; j<layer->n_neurons; j++)
    {
        tmp = get_neuron_output(layer->neurons[j], inputs);
        activated = sigmoid(tmp);
        output[j] = activated;
        layer->neurons[j]->activated_output = activated;
    }
    return output; // rmb to free memory
}

NETWORK * init_network(const int n_features, const int n_classes, const int n_layers, const int units[], const double learning_rate)
{
    NETWORK * network = (NETWORK*)malloc(sizeof(NETWORK));
    network->n_layers = n_layers;
    network->n_features = n_features;
    network->n_classes = n_classes;
    network->learning_rate = learning_rate;
    network->layers = (LAYER**)malloc(sizeof(LAYER*)*n_layers);
    int j;
    for (j=0; j<n_layers; j++)
    {
        network->layers[j] = init_layer(j, (j!=n_layers-1)?units[j]:n_classes, (j==0)?n_features:units[j-1]);
    }
    return network;
}

void release_network(NETWORK * network)
{
    int j;
    for (j=0; j<network->n_layers; j++)
    {
        release_layer(network->layers[j]);
    }
    free(network);
}

void print_network_weights(NETWORK * network)
{
    int j;
    for (j=0; j<network->n_layers; j++)
    {
        print_layer_weights(network->layers[j]);
    }
}

double * feed_forward(NETWORK * network, const double *inputs)
{
    double * tmp = (double*)malloc(sizeof(double)*network->n_features);
    int j, k;
    for (j=0; j<network->n_features; j++)
        tmp[j] = inputs[j];

    for (j=0; j<network->n_layers; j++)
    {
        double * tmptmp = get_layer_output(network->layers[j], tmp);
        free(tmp);
        tmp = (double*)malloc(sizeof(double)*network->layers[j]->n_neurons);
        for (k=0; k<network->layers[j]->n_neurons; k++)
            tmp[k] = tmptmp[k];
        free(tmptmp);
    }
    return tmp;
}

void feed_backward(NETWORK * network, const int *labels)
{
    //printf("\nfeed backward begin\n");
    int j;
    LAYER * lastlayer = network->layers[network->n_layers-1];
    for (j=0; j<network->n_classes; j++)
    {
        double error = -(labels[j]-lastlayer->neurons[j]->activated_output);
        lastlayer->neurons[j]->delta = error * d_sigmoid(lastlayer->neurons[j]->activated_output);
    }
    int l = network->n_layers - 2;
    while (l >= 0)
    {
        LAYER * clayer = network->layers[l];
        LAYER * player = network->layers[l+1];
        //printf("processing layer: %d\n", clayer->idx);
        // update current layer
        int k;
        for (k=0; k<clayer->n_neurons; k++)
        {
            double acc_error = 0.0;
            int m;
            for (m=0; m<player->n_neurons; m++)
            {
                acc_error += player->neurons[m]->delta * player->neurons[m]->weights[k];
            }
            clayer->neurons[k]->delta = acc_error * d_sigmoid(clayer->neurons[k]->activated_output);
        }
        l -= 1;
    }
    //printf("feed backward completed\n");
}

void update_neuron_weights(NEURON * neuron, const double learning_rate)
{
    int j;
    for (j=0; j<neuron->n_weights; j++)
    {
        neuron->weights[j] -= learning_rate * neuron->delta * neuron->inputs[j];
    }
    neuron->bias -= learning_rate * neuron->delta;
    neuron->output = get_neuron_output(neuron, neuron->inputs); // query the needs for this line
}

void update_layer_weights(LAYER * layer, const double learning_rate)
{
    int j;
    for (j=0; j<layer->n_neurons; j++)
    {
        update_neuron_weights(layer->neurons[j], learning_rate);
    }
}

void update_network_weights(NETWORK * network)
{
    int j;
    for (j=0; j<network->n_layers; j++)
    {
        update_layer_weights(network->layers[j], network->learning_rate);
    }
}

double loss_function(const int *ytrue, const double *yhat, const int n)
{
    int j;
    double acc_loss = 0.0;
    for (j=0; j<n; j++)
    {
        acc_loss += pow(ytrue[j]-yhat[j], 2);
    }
    return acc_loss;
}

double get_total_error(NETWORK * network, double **list_of_inputs, int **list_of_labels, const int n, int print)
{
    double acc_error = 0.0;
    int j;
    for (j=0; j<n; j++)
    {
        double *yhat = feed_forward(network, list_of_inputs[j]);
        acc_error += loss_function(list_of_labels[j], yhat, network->n_classes);
        if (print)
        {
            int k;
            for (k=0; k<3; k++)
            {
                printf("%lf ", yhat[k]);
            }
            printf("\tvs\t");
            for (k=0; k<3; k++)
            {
                printf("%d ", list_of_labels[j][k]);
            }
            puts("");
        }
        free(yhat);
    }
    return acc_error / n;
}

void fit(NETWORK * network, double **list_of_features, int **list_of_labels, 
        const int n, const int max_iter)
{
    int j, k, iter;
    for (iter=0; iter<max_iter; iter++)
    {
        printf("Iteration %d:\n", iter);
        for (j=0; j<n; j++)
        {
            // printf("j %d:\n", j);
            double *yhat = feed_forward(network, list_of_features[j]);
            // printf("feedforward\n");
            feed_backward(network, list_of_labels[j]);
            // printf("feedbackward\n");
            update_network_weights(network);
            // printf("update weight\n");
            free(yhat);
            // printf("free yhat\n");
        }
        double error = get_total_error(network, list_of_features, list_of_labels,n, false);
        printf("Iteration: %d\nLoss: %.6lf\n", iter, error);
    }
}

double **min_max_scale(double **data, const int n_rows, const int n_columns)
{
    double **output = (double**)malloc(sizeof(double*)*n_rows);
    double *column_min = (double*)malloc(sizeof(double)*n_columns);
    double *column_max = (double*)malloc(sizeof(double)*n_columns);
    int j, k;
    for (j=0; j<n_columns; j++)
    {
        column_min[j] = __DBL_MAX__;
        column_max[j] = __DBL_MIN__;
    }
    for (j=0; j<n_rows; j++)
    {
        output[j] = (double*)malloc(sizeof(double)*n_columns);
        for (k=0; k<n_columns; k++)
        {
            if (data[j][k] < column_min[k])
            {
                column_min[k] = data[j][k];
            }
            if (data[j][k] > column_max[k])
            {
                column_max[k] = data[j][k];
            }
        }
    }
    for (j=0; j<n_rows; j++)
    {
        for (k=0; k<n_columns; k++)
        {
            output[j][k] = (data[j][k]-column_min[k]) / (column_max[k]-column_min[k]);
        }
    }
    free(column_min);
    free(column_max);
    return output;
}
int summation(int input[], int n)
{
    int total = 0;
    return total;
}