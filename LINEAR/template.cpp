#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "data_loader.hpp"
#include "node.h"
#include "kernel.h"

using namespace std;

int no_of_nodes;
int edge_list_size;

Node* h_graph_nodes;
int* h_graph_edges;

double* y_array;
double* x_array;
double* weight_old;
double* weight;
double* weight_ind;

int dim_num;
int max_iters;
int num_threads;
int mode;
int batch_size;
int stale_mode;
int stale_threshold;

double norm(double* weight, double* weight_old, int dim_num){
    double sum = 0;
    for (size_t i = 0; i < dim_num; ++i){
        double minus = weight[i] - weight_old[i];
        double r = minus * minus;
        sum += r;
    }   

    return sqrt(sum);
}

void compute(){
  double epsilon = 0.001; // the convergence rate

  int iter = 0;
  int num_batches = no_of_nodes / batch_size;
  if(num_batches * batch_size < no_of_nodes){
    num_batches++;
  }

  for (size_t k = 0; k < dim_num; ++k) {
    weight_old[k] = weight[k];
  }

  bool best_found = false;
  double dist;
  while (true) {
    for(int batch_id = 0; batch_id < num_batches; batch_id++){
      best_found = false;
      
      kernel(batch_id);

      if(mode == 1){
        if(batch_id == num_batches - 1){
          dist = norm(weight, weight_old, dim_num);
        }
        for (size_t k = 0; k < dim_num; ++k) {
          weight_old[k] = weight[k];
        }
      }      

      if(stale_mode == 1){
        int next_batch_id = batch_id + stale_threshold;
        if(next_batch_id >= num_batches){
          next_batch_id = next_batch_id % num_batches;
        }
        Sync_batch(next_batch_id);  
      }
    }

    if(mode == 0){ 
      dist = norm(weight, weight_old, dim_num);
      for (size_t k = 0; k < dim_num; ++k) {
        weight_old[k] = weight[k];
      }
    }
    
    cout << dist << endl;
    if(dist < epsilon){
      cout << "best_weight" << endl;
      best_found = true;
    }

    if(best_found == true){
      break;
    }

    iter++;
    if (iter >= max_iters) {
      break;
    }
  }

  cout << "iteration: " << iter << endl;    
}

int main(int argc, char* argv[]) {
    no_of_nodes = atoi(argv[2]);
    edge_list_size = 0;
    dim_num = atoi(argv[3]);
    max_iters = atoi(argv[4]);
    num_threads = atoi(argv[5]);
    mode = atoi(argv[6]);
    batch_size = atoi(argv[7]);
    stale_mode = atoi(argv[8]);
    stale_threshold = atoi(argv[9]);

    h_graph_nodes = (Node*) calloc(no_of_nodes, sizeof(Node));    
    for(int i = 0; i < no_of_nodes; i++){
        h_graph_nodes[i].starting = 0;
        h_graph_nodes[i].no_of_edges = 0;
    }

    string input_name = "../INPUT_ML/"; 
    input_name += string(argv[1]);  

    SimpleDataLoader loader;
    loader.setup_nodes(input_name.c_str(), h_graph_nodes);
    for(int i = 0; i < no_of_nodes; i++){
        h_graph_nodes[i].starting = edge_list_size;
        edge_list_size += h_graph_nodes[i].no_of_edges;
    }
    h_graph_edges = (int*) calloc(edge_list_size, sizeof(int));
    y_array = (double*) calloc(no_of_nodes, sizeof(double));
    x_array = (double*) calloc(edge_list_size, sizeof(double));
    weight_old = (double*)calloc(dim_num, sizeof(double));
    weight = (double*)calloc(dim_num, sizeof(double));
    weight_ind = (double*)calloc(edge_list_size, sizeof(double));
    
    loader.load_file(input_name.c_str(), y_array, x_array, h_graph_nodes, h_graph_edges);

    cout << "Execution Start"<< endl;    

    // lr_method
    compute();
    return 0;
}
