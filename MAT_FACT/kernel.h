#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstring>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "node.h"

extern int row_count;
extern int col_count; // Number of rows and cols
extern int K_count;
extern int num_iterations;

extern int no_of_nodes;

extern int* x_row_array;
extern int* x_col_array;
extern double* x_val_array;

// PS Table
extern double* L_table;
extern double* R_table;
extern double* R_table_ind;
extern double loss_sum;

extern int num_threads;
extern int mode;
extern int batch_size;
extern int stale_mode;
extern int stale_threshold;

bool atomicAdd(double* address, double val);
void Sync_batch(int batch_id);
void kernel(int iter, int batch_id);