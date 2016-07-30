#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include "data_loader.hpp"
#include "node.h"
#include "kernel.h"

using namespace std;
typedef numeric_limits< double > dbl;

// Data variables
int row_count;
int col_count; // Number of rows and cols
int K_count;
int num_iterations;

int no_of_nodes;

int* x_row_array;
int* x_col_array;
double* x_val_array;

// PS Table
double* L_table;
double* R_table;
double* R_table_ind;
double loss_sum;

int num_threads;
int mode;
int batch_size;
int stale_mode;
int stale_threshold;

// Initialize the Matrix Factorization solver
void init_mf(double* L_table, double* R_table) {
    // Create a uniform RNG in the range [0,1)
    int rng_seed =  967234;
    boost::mt19937 generator(rng_seed);
    boost::uniform_real<> zero_to_one_dist(0,1);
    boost::variate_generator<boost::mt19937&,boost::uniform_real<> > zero_to_one_generator(generator, zero_to_one_dist);
    
    // Add a random initialization in [0,1)/num_workers to each element of L and R
    // L_table (N by K)
    for (int i = 0; i < row_count; ++i) {
      for (int k = 0; k < K_count; ++k) {
        double value = zero_to_one_generator();
        double init_val = (2.0 * value - 1.0);
        L_table[(i * K_count) + k] +=init_val;
      }
    }
    
    // R_table (M by K)
    for (int j = 0; j < col_count; ++j) {
      for (int k = 0; k < K_count; ++k) {
        double value = zero_to_one_generator();
        double init_val = (2.0 * value - 1.0);
        R_table[(j * K_count) + k] +=init_val;
      }
    }
}

void compute(){
  int iter = 1;
  int num_batches = no_of_nodes / batch_size;
  if(num_batches * batch_size < no_of_nodes){
    num_batches++;
  }

  while (true) {
    loss_sum = 0;

    for(int batch_id = 0; batch_id < num_batches; batch_id++){
      kernel(iter, batch_id);
      if(stale_mode == 1){
        int next_batch_id = batch_id + stale_threshold;
        if(next_batch_id >= num_batches){
          next_batch_id = next_batch_id % num_batches;
        }
        Sync_batch(next_batch_id);  
      }      
    }

    cout << (double)loss_sum << endl;
    iter++;
    if (iter > num_iterations) {
      break;
    }
  }
  cout << "iteration: " << iter << endl;  
}

// Main function
int main(int argc, char *argv[]) {
  string input_name = "../INPUT_ML/"; 
  input_name += string(argv[1]);  

  // Load Data
  SimpleDataLoader loader;
  no_of_nodes = loader.load_file(input_name.c_str(), x_row_array, x_col_array, x_val_array);

  row_count = atoi(argv[2]);
  col_count = atoi(argv[3]);
  K_count = atoi(argv[4]);
  num_iterations = atoi(argv[5]);
  num_threads = atoi(argv[6]);
  mode = atoi(argv[7]);
  batch_size = atoi(argv[8]);
  stale_mode = atoi(argv[9]);
  stale_threshold = atoi(argv[10]);
  
  // Configure table
  L_table = (double*)malloc((row_count * K_count) * sizeof(double));
  for(int i = 0; i < (row_count * K_count); i++){
    L_table[i] = 0;
  } 
  R_table = (double*)malloc((col_count * K_count) * sizeof(double));
  for(int i = 0; i < (col_count * K_count); i++){
    R_table[i] = 0; 
  } 

  init_mf(L_table, R_table);

  R_table_ind = (double*)malloc((no_of_nodes * K_count) * sizeof(double));
  for( unsigned int i = 0; i < no_of_nodes; i++) {
    int column_index = x_col_array[i];
    for (int k = 0; k < K_count; ++k) {
      R_table_ind[(i * K_count) + k] = R_table[(column_index * K_count) + k];
    }
  }

  cout << "Execution Start" << endl;  

  // Run MF solver
  compute();
  
  return 0;
}
