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
    double *delta;
    double **inputs;
    double *activated_outputs;
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
    int batch_size;
    LAYER **layers;
    double learning_rate;
} NETWORK;

void release_data(void **data, const int n_rows)
{
    int j;
    for (j = 0; j < n_rows; j++)
        free(data[j]);
    free(data);
}

// need attention on the initializer
NEURON *init_neuron(const int i, const int in, const int out, const int batch_size)
{
    NEURON *neuron = (NEURON *)malloc(sizeof(NEURON));
    neuron->idx = i;
    neuron->n_weights = in;
    neuron->weights = (double *)malloc(sizeof(double) * in);
    double limit = sqrt(6 / (in + out));
    printf("#%lf", limit);
    neuron->bias = (rand() % 860000) * 0.000001 * (rand() % 2 == 0 ? 1 : -1);
    neuron->delta = (double *)malloc(sizeof(double) * batch_size);
    neuron->inputs = (double **)malloc(sizeof(double *) * batch_size);
    int j;
    for (j = 0; j < batch_size; j++)
        neuron->inputs[j] = (double *)malloc(sizeof(double) * in);
    neuron->activated_outputs = (double *)malloc(sizeof(double) * batch_size);
    for (j = 0; j < in; j++)
    {
        neuron->weights[j] = (rand() % 860000) * 0.000001 * (rand() % 2 == 0 ? 1 : -1);
    }

    return neuron;
}

void release_neuron(NEURON *neuron, const int batch_size)
{
    free(neuron->weights);
    release_data((void **)neuron->inputs, batch_size);
    free(neuron->delta);
    free(neuron->activated_outputs);
    free(neuron);
}

void print_neuron_weights(NEURON *neuron)
{
    int j;
    printf("(");
    for (j = 0; j < neuron->n_weights; j++)
        printf("%.6lf,", neuron->weights[j]);
    printf(")");
}

double *get_neuron_outputs(NEURON *neuron, double **inputs, const int batch_size)
{
    int j, k;
    for (j = 0; j < batch_size; j++)
    {
        //neuron->inputs[j] = (double *)malloc(sizeof(double) * neuron->n_weights);
        double tmp = 0.0;
        for (k = 0; k < neuron->n_weights; k++)
        {
            neuron->inputs[j][k] = inputs[j][k];
            tmp += neuron->weights[j] * inputs[j][k];
        }
        neuron->activated_outputs[j] = sigmoid(tmp + neuron->bias);
    }
    return neuron->activated_outputs;
}

LAYER *init_layer(const int i, const int n_neurons, const int n_features, const int out, const int batch_size)
{
    LAYER *layer = (LAYER *)malloc(sizeof(LAYER));
    layer->idx = i;
    layer->n_neurons = n_neurons;
    layer->neurons = (NEURON **)malloc(sizeof(NEURON *) * n_neurons);
    int j;
    for (j = 0; j < n_neurons; j++)
    {
        layer->neurons[j] = init_neuron(j, n_features, out, batch_size);
    }
    return layer;
}

void release_layer(LAYER *layer, const int batch_size)
{
    int j;
    for (j = 0; j < layer->n_neurons; j++)
    {
        release_neuron(layer->neurons[j], batch_size);
    }
    free(layer);
}

void print_layer_weights(LAYER *layer)
{
    int j;
    printf("\nlayer: %d\n", layer->idx);
    printf("[");
    for (j = 0; j < layer->n_neurons; j++)
        print_neuron_weights(layer->neurons[j]);
    printf("]");
}

double **get_layer_output(LAYER *layer, double **inputs, const int batch_size)
{
    double **output = (double **)malloc(sizeof(double *) * batch_size);
    double *yhat;
    int j, k, l;
    for (k = 0; k < layer->n_neurons; k++)
    {
        yhat = get_neuron_outputs(layer->neurons[k], inputs, batch_size);
    }
    for (j = 0; j < batch_size; j++)
    {
        output[j] = (double *)malloc(sizeof(double) * layer->n_neurons);
        for (k = 0; k < layer->n_neurons; k++)
        {
            output[j][k] = layer->neurons[k]->activated_outputs[j];
        }
    }
    return output; // rmb to free memory
}

NETWORK *init_network(const int n_features, const int n_classes, const int n_layers, const int units[], const double learning_rate,
                      const int batch_size)
{
    NETWORK *network = (NETWORK *)malloc(sizeof(NETWORK));
    network->n_layers = n_layers;
    network->n_features = n_features;
    network->n_classes = n_classes;
    network->batch_size = batch_size;
    network->learning_rate = learning_rate;
    network->layers = (LAYER **)malloc(sizeof(LAYER *) * n_layers);
    int j;
    for (j = 0; j < n_layers; j++)
    {
        network->layers[j] = init_layer(j, (j != n_layers - 1) ? units[j] : n_classes, (j == 0) ? n_features : units[j - 1], (j != n_layers - 1) ? units[j + 1] : n_classes, batch_size);
    }
    return network;
}

void release_network(NETWORK *network)
{
    int j;
    for (j = 0; j < network->n_layers; j++)
    {
        release_layer(network->layers[j], network->batch_size);
    }
    free(network);
}

void print_network_weights(NETWORK *network)
{
    int j;
    for (j = 0; j < network->n_layers; j++)
    {
        print_layer_weights(network->layers[j]);
    }
}

double **feed_forward(NETWORK *network, double **inputs, const int batch_size)
{
    double **tmp = (double **)malloc(sizeof(double *) * batch_size);
    int j, k, l;
    // preparation
    //printf("preparation\n");
    for (j = 0; j < batch_size; j++)
    {
        tmp[j] = (double *)malloc(sizeof(double) * network->n_features);
        for (k = 0; k < network->n_features; k++)
        {
            tmp[j][k] = inputs[j][k];
        }
    }
    // feed
    //printf("Feeding\n");
    for (j = 0; j < network->n_layers; j++)
    {
        if (j != network->n_layers)
        {
            double **tmptmp = get_layer_output(network->layers[j], tmp, batch_size);
            //printf("Fed\n");
            release_data((void **)tmp, batch_size);
            //printf("Released\n");
            tmp = (double **)malloc(sizeof(double *) * batch_size);
            for (k = 0; k < batch_size; k++)
            {
                tmp[k] = (double *)malloc(sizeof(double) * (network->layers[j]->n_neurons));
                //printf("allocated %d %d\n", j, network->layers[j]->n_neurons);
                //printf("test: %lf\n", tmptmp[k][0]);
                for (l = 0; l < network->layers[j]->n_neurons; l++)
                {
                    tmp[k][l] = tmptmp[k][l];
                    //printf("%lf\t", tmptmp[k][l]);
                }
                //puts("");
            }
            //printf("Copied\n");
            /* 
            only release the memory allocated for the top layer will do,
            inner cell take references from neuron
            */
            release_data((void **)tmptmp, network->batch_size);
        }
        else
        {
            // softmax activation
            getchar();
            double tmp_total = 0.0;
            for (k = 0; k < batch_size; k++)
            {
                //tmp[k] = (double *)malloc(sizeof(double) * (network->layers[j]->n_neurons));
                for (l = 0; l < network->layers[j]->n_neurons; l++)
                {
                    tmp[k][l] = exp(tmp[k][l]);
                    tmp_total += tmp[k][l];
                    //network->layers[j]->neurons[l]->activated_outputs[k] = tmp[k][l];
                }
                for (l = 0; l < network->layers[j]->n_neurons; l++)
                {
                    network->layers[j]->neurons[l]->activated_outputs[k] = tmp[k][l] / tmp_total;
                }
            }
        }
    }
    return tmp;
}

void feed_backward(NETWORK *network, int **labels, const int batch_size)
{
    //printf("\nfeed backward begin\n");
    int j, k, l = network->n_layers - 2, m;
    LAYER *lastlayer = network->layers[network->n_layers - 1];
    // query input of d_sigmoid should be inputs[i][j]
    for (j = 0; j < batch_size; j++)
    {
        for (k = 0; k < network->n_classes; k++)
        {
            double error = -(labels[j][k] - lastlayer->neurons[k]->activated_outputs[j]);
            lastlayer->neurons[k]->delta[j] = error * d_sigmoid(lastlayer->neurons[k]->activated_outputs[j]);
        }
    }

    while (l >= 0)
    {
        LAYER *clayer = network->layers[l];
        LAYER *player = network->layers[l + 1];
        //printf("processing layer: %d\n", clayer->idx);
        // update current layer
        for (j = 0; j < batch_size; j++)
        {
            for (k = 0; k < clayer->n_neurons; k++)
            {
                double acc_error = 0.0;
                for (m = 0; m < player->n_neurons; m++)
                {
                    acc_error += player->neurons[m]->delta[j] * player->neurons[m]->weights[k];
                }
                clayer->neurons[k]->delta[j] = acc_error * d_sigmoid(clayer->neurons[k]->activated_outputs[j]);
            }
        }
        l -= 1;
    }
    //printf("feed backward completed\n");
}

double reduce_sum(double *delta, const int batch_size)
{
    int j;
    double acc = 0.0;
    for (j = 0; j < batch_size; j++)
        acc += delta[j];
    return acc / batch_size;
}

void update_neuron_weights(NEURON *neuron, const double learning_rate, const int batch_size)
{
    int j, k;
    double batch_delta;
    double *tmp = (double *)malloc(sizeof(double) * batch_size);
    for (j = 0; j < neuron->n_weights; j++)
    {
        for (k = 0; k < batch_size; k++)
        {
            tmp[k] = neuron->delta[k] * neuron->inputs[k][j];
        }
        neuron->weights[j] -= learning_rate * reduce_sum(tmp, batch_size);
    }
    batch_delta = reduce_sum(neuron->delta, batch_size);
    neuron->bias -= learning_rate * batch_delta;
    /*
    return value are an active reference, therefore no release required
    */
    //get_neuron_outputs(neuron, neuron->inputs, batch_size);
    free(tmp);
}

void update_layer_weights(LAYER *layer, const double learning_rate, const int batch_size)
{
    int j;
    for (j = 0; j < layer->n_neurons; j++)
    {
        update_neuron_weights(layer->neurons[j], learning_rate, batch_size);
    }
}

void update_network_weights(NETWORK *network)
{
    int j;
    for (j = 0; j < network->n_layers; j++)
    {
        update_layer_weights(network->layers[j], network->learning_rate, network->batch_size);
    }
}

double loss_function(int **ytrue, double **yhat, const int n, const int batch_size)
{
    int j, k;
    double acc_loss = 0.0;
    for (j = 0; j < batch_size; j++)
    {
        for (k = 0; k < n; k++)
            acc_loss += pow(ytrue[j][k] - yhat[j][k], 2);
    }
    return acc_loss;
}

double **extract_batch_double(double **inputs, const int col, const int batch_size, const int pos)
{
    double **batch_inputs = (double **)malloc(sizeof(double *) * batch_size);
    int j, k, p = pos * batch_size;
    for (j = 0; j < batch_size; j++)
    {
        //printf("%d -> %d\n", p, j);
        batch_inputs[j] = (double *)malloc(sizeof(double) * col);
        for (k = 0; k < col; k++)
        {
            batch_inputs[j][k] = inputs[p][k];
        }
        p++;
    }
    return batch_inputs;
}

int **extract_batch_int(int **inputs, const int col, const int batch_size, const int pos)
{
    int **batch_inputs = (int **)malloc(sizeof(int *) * batch_size);
    int j, k, p = pos * batch_size;
    for (j = 0; j < batch_size; j++)
    {
        batch_inputs[j] = (int *)malloc(sizeof(int) * col);
        for (k = 0; k < col; k++)
        {
            batch_inputs[j][k] = inputs[p][k];
        }
        p++;
    }
    return batch_inputs;
}

double get_total_error(NETWORK *network, double **list_of_inputs, int **list_of_labels, const int n, int print)
{
    double acc_error = 0.0;
    double **batch_inputs;
    int **batch_labels;
    int j;
    int loop = n / network->batch_size;
    int rem = n % network->batch_size; // take note!
    for (j = 0; j < loop; j++)
    {
        batch_inputs = extract_batch_double(list_of_inputs, network->n_features, network->batch_size, j);
        batch_labels = extract_batch_int(list_of_labels, network->n_classes, network->batch_size, j);
        double **yhat = feed_forward(network, batch_inputs, network->batch_size);
        acc_error += loss_function(batch_labels, yhat, network->n_classes, network->batch_size);
        if (print)
        {
            int k, l;
            for (k = 0; k < network->batch_size; k++)
            {
                for (l = 0; l < network->n_classes; l++)
                {
                    printf("%lf\t", yhat[k][l]);
                }
                printf("VS\t");
                for (l = 0; l < network->n_classes; l++)
                    printf("%d\t", batch_labels[k][l]);
                puts("");
            }
        }
        release_data((void **)batch_inputs, network->batch_size);
        release_data((void **)batch_labels, network->batch_size);
        release_data((void **)yhat, network->batch_size);
    }
    return acc_error / (n - rem);
}

double get_total_accuracy(NETWORK *network, double **list_of_inputs, int **list_of_labels, const int n)
{
    double hit = 0;
    double **batch_inputs;
    int **batch_labels;
    int j, k, l;
    int loop = n / network->batch_size;
    int rem = n % network->batch_size; // take note!
    for (j = 0; j < loop; j++)
    {
        batch_inputs = extract_batch_double(list_of_inputs, network->n_features, network->batch_size, j);
        batch_labels = extract_batch_int(list_of_labels, network->n_classes, network->batch_size, j);
        double **yhat = feed_forward(network, batch_inputs, network->batch_size);
        for (k = 0; k < network->batch_size; k++)
        {
            double highest = -1;
            int highest_idx = -1;
            for (l = 0; l < network->n_classes; l++)
            {
                if (yhat[k][l] > highest)
                {
                    highest = yhat[k][l];
                    highest_idx = l;
                }
            }
            highest = -1;
            int truth = -1;
            for (l = 0; l < network->n_classes; l++)
            {
                if (batch_labels[k][l] > highest)
                {
                    highest = batch_labels[k][l];
                    truth = l;
                }
            }
            printf("%d vs %d\n", highest_idx, truth);
            if (highest_idx == truth)
            {
                if (truth == -1)
                    printf("Oh No!\n");
                else
                {
                    printf("%d\n", truth);
                }
                hit += 1;
            }
        }
        release_data((void **)batch_inputs, network->batch_size);
        release_data((void **)batch_labels, network->batch_size);
        release_data((void **)yhat, network->batch_size);
    }
    printf("n: %d, loop: %d\n", (n-rem), loop);
    return hit / (n - rem);
}
int *get_random_sequence(const int len)
{
    int *seq = (int *)malloc(sizeof(int) * len);
    int i, j, tmp, insert;
    for (i = 0; i < len;)
    {
        tmp = rand() % len;
        insert = 1;
        for (j = 0; j < i; j++)
        {
            if (seq[j] == tmp)
            {
                insert = 0;
                break;
            }
        }
        if (insert)
            seq[i++] = tmp;
    }
    return seq;
}

double **map_seq2index(double **list_of_features, int *seq, const int n, const int c)
{
    double **new_seq = (double **)malloc(sizeof(double *) * n);
    int i, j;
    for (i = 0; i < n; i++)
    {
        new_seq[i] = (double *)malloc(sizeof(double) * c);
        for (j = 0; j < c; j++)
        {
            new_seq[i][j] = list_of_features[seq[i]][j];
        }
    }
    return new_seq;
}

int **map_seq2_index(int **list_of_labels, int *seq, const int n, const int c)
{
    int **new_seq = (int **)malloc(sizeof(int *) * n);
    int i, j;
    for (i = 0; i < n; i++)
    {
        new_seq[i] = (int *)malloc(sizeof(int) * c);
        for (j = 0; j < c; j++)
        {
            new_seq[i][j] = list_of_labels[seq[i]][j];
        }
    }
    return new_seq;
}

void fit(NETWORK *network, double **list_of_features, int **list_of_labels,
         const int n, const int max_iter)
{
    int j, k, iter;
    double **batch_inputs;
    int **batch_labels;
    int loop = n / network->batch_size;
    int rem = n % network->batch_size; // take note!
    for (iter = 0; iter < max_iter; iter++)
    {
        printf("Iteration %d:\n", iter);
        int *seq = get_random_sequence(n);
        double **shuffled_features = map_seq2index(list_of_features, seq, n, 5);
        int **shuffled_labels = map_seq2_index(list_of_labels, seq, n, 5);
        for (j = 0; j < loop; j++)
        {
            batch_inputs = extract_batch_double(shuffled_features, network->n_features, network->batch_size, j);
            batch_labels = extract_batch_int(shuffled_labels, network->n_classes, network->batch_size, j);
            //printf("j %d:\n", j);
            double **yhat = feed_forward(network, batch_inputs, network->batch_size);
            //printf("feedforward\n");
            feed_backward(network, batch_labels, network->batch_size);
            //printf("feedbackward\n");
            update_network_weights(network);
            //printf("update weight\n");
            release_data((void **)yhat, network->batch_size);
            //printf("free yhat\n");
            release_data((void **)batch_inputs, network->batch_size);
            release_data((void **)batch_labels, network->batch_size);
        }
        free(seq);
        release_data((void **)shuffled_features, n);
        release_data((void **)shuffled_labels, n);
        double error = get_total_error(network, list_of_features, list_of_labels, n, false);
        printf("Iteration: %d\nLoss: %.6lf\n", iter, error);
    }
}

double **batchwise_feed_forward(double **x, const int start_idx, const int end_idx, const int batch_size)
{
    double **output = (double **)malloc(sizeof(double *) * batch_size);

    return output;
}

double **min_max_scale(double **data, const int n_rows, const int n_columns)
{
    double **output = (double **)malloc(sizeof(double *) * n_rows);
    double *column_min = (double *)malloc(sizeof(double) * n_columns);
    double *column_max = (double *)malloc(sizeof(double) * n_columns);
    int j, k;
    for (j = 0; j < n_columns; j++)
    {
        column_min[j] = __DBL_MAX__;
        column_max[j] = __DBL_MIN__;
    }
    for (j = 0; j < n_rows; j++)
    {
        output[j] = (double *)malloc(sizeof(double) * n_columns);
        for (k = 0; k < n_columns; k++)
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
    for (j = 0; j < n_rows; j++)
    {
        for (k = 0; k < n_columns; k++)
        {
            output[j][k] = (data[j][k] - column_min[k]) / (column_max[k] - column_min[k]);
        }
    }
    free(column_min);
    free(column_max);
    return output;
}
