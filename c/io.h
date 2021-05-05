#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define COUNT 128

double *data_tok(char *data, char *tok, const int n_columns)
{
    double *result = (double *)malloc(sizeof(double) * n_columns);
    //char **args = (char**)malloc(sizeof(char*)*n_columns);
    int idx = 0;
    char *token = strtok(data, tok);
    while (token != NULL)
    {
        result[idx] = strtod(token, NULL);
        //printf("%3.1lf\t", result[idx]);
        idx++;
        token = strtok(NULL, tok);
    }
    return result;
}

double **get_data(const char *path, const int n_rows, const int n_columns, char tok[])
{
    double **result = (double **)malloc(sizeof(double *) * n_rows);
    FILE *freader = fopen(path, "r");
    int j, k, row_counter = 0;
    for (j = 0; j < n_rows; j++)
    {
        char tmp[COUNT] = {0};
        fgets(tmp, COUNT, freader);
        double *record = data_tok(tmp, tok, n_columns);
        result[j] = record;
        //puts("");
    }
    fclose(freader);
    return result;
}
/*
void release_data(void **data, const int n_rows)
{
    int j;
    for (j = 0; j < n_rows; j++)
        free(data[j]);
    free(data);
}
*/
double *get_by_column(double **data, const int col_idx, const int n_rows)
{
    double *output = (double *)malloc(sizeof(double) * n_rows);
    int j;
    for (j = 0; j < n_rows; j++)
        output[j] = data[j][col_idx];
    return output;
}

double **drop_column(double **data, const int col_idx, const int n_rows, const int n_columns)
{
    double **output = (double **)malloc(sizeof(double *) * n_rows);
    int i, j, k;
    for (j = 0; j < n_rows; j++)
    {
        output[j] = (double *)malloc(sizeof(double) * n_columns - 1);
        i = 0;
        for (k = 0; k < n_columns; k++)
        {
            if (k != col_idx)
            {
                output[j][i] = data[j][k];
                i++;
            }
        }
    }
    return output;
}

int *double_to_int(double *data, const int n_rows)
{
    int *output = (int *)malloc(sizeof(int) * n_rows);
    int j;
    for (j = 0; j < n_rows; j++)
    {
        output[j] = (int)data[j];
    }
    return output;
}
int **one_hot_encoding(double *data, const int n_rows, const int n_classes)
{
    int *int_data = double_to_int(data, n_rows);
    int **output = (int **)malloc(sizeof(int *) * n_rows);
    int j, k;
    for (j = 0; j < n_rows; j++)
    {
        output[j] = (int *)malloc(sizeof(int) * n_classes);
        for (k = 0; k < n_classes; k++)
            output[j][k] = 0;
        output[j][int_data[j]-1] = 1;
    }
    return output;
}
/*
int main(void)
{
    const char * file_path = ".//Final Project - thyroid//testing_data1.txt"; // 
    const int n_columns = 6;
    const int n_rows = 108;
    const int labels_idx = 5;
    char tok[2] = "\t";
    double **data = get_data(file_path, n_rows, n_columns, tok);
    release_data(data, n_rows);
    return 0;
}
*/