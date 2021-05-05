#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nn_mini.h"
#include "io.h"

int main(void)
{
    srand(time(NULL));
    printf("%s", "Hello World\n");
    const char *file_path = ".//Final Project - thyroid//training_data2.txt"; //
    char tok[2] = "\t";
    const int n_columns = 6;
    const int n_rows = 100;
    const int labels_idx = 5;
    const int n_classes = 3;
    const int batch_size = 100;
    int units[] = {128};
    NETWORK *network = init_network(n_columns - 1, n_classes, sizeof(units) / sizeof(units[0]) + 1, units, 0.05, batch_size);
    printf("learning rate: %lf\n", network->learning_rate);
    print_network_weights(network);
    getchar();
    /*
    Train Data
    */
    double **data = get_data(file_path, n_rows, n_columns, tok);
    int k, l;
    double *labels = get_by_column(data, labels_idx, n_rows); // rmb to release!
    int **n_labels = one_hot_encoding(labels, n_rows, n_classes);
    double **n_features = drop_column(data, labels_idx, n_rows, n_columns);
    double **scaled_features = min_max_scale(n_features, n_rows, n_columns);
    puts("PRE-FIT");
    /*
    for (k=0; k<n_rows; k++)
    {
        for (l=0; l<n_columns-1; l++)
            printf("%lf\t", scaled_features[k][l]);
        puts("");
    }
    */
    fit(network, scaled_features, n_labels, n_rows, 10000);
    puts("POST-FIT");
    /*
    Test Data
    */
    // utilize config file
    //const char *test_file = ".//Final Project - thyroid//training_data1.txt"; //
    const char *test_file = ".//Final Project - thyroid//testing_data2.txt"; //
    double **test_data = get_data(test_file, 100, n_columns, tok);
    double *test_labels = get_by_column(test_data, labels_idx, 100); // rmb to release!
    int **test_encoded_labels = one_hot_encoding(test_labels, 100, n_classes);
    double **test_features = drop_column(test_data, labels_idx, 100, n_columns);
    double **scaled_test_features = min_max_scale(test_features, 100, n_columns);

    int *seq = get_random_sequence(100);
    double **shuffled_features = map_seq2index(scaled_test_features, seq, 100, 5);
    int **shuffled_labels = map_seq2_index(test_encoded_labels, seq, 100, 5);
    double error = get_total_error(network, shuffled_features, shuffled_labels, 100, true);
    double accuracy = get_total_accuracy(network, shuffled_features, shuffled_labels, 100);
    release_data((void**)shuffled_labels, 100);
    release_data((void**)shuffled_features, 100);
    free(seq);
    printf("\n\nERROR: %lf\n\n", error);
    printf("\n\nAccuracy: %lf\n\n", accuracy);
    print_network_weights(network);
    puts("RELEASING TRAIN DATA");
    release_network(network);
    release_data((void **)data, n_rows);
    free(labels);
    release_data((void **)n_features, n_rows);
    release_data((void **)n_labels, n_rows);
    release_data((void **)scaled_features, n_rows);

    // test section
    puts("RELEASING TEST DATA");
    release_data((void **)test_data, 100);
    release_data((void **)test_features, 100);
    release_data((void **)test_encoded_labels, 100);
    release_data((void **)scaled_test_features, 100);
    free(test_labels);
    return 0;
}
/*
    s = 1 / (1 + np.exp(-Z))
    return s * (1 - s)
*/