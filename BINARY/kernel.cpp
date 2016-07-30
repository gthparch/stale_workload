#include "kernel.h"

double sigmoid(double x) {
    double e = 2.718281828;
    return 1.0 / (1.0 + pow(e, -x));
}

bool atomicAdd(double* address, double val){    
    uint64_t* address_uint = reinterpret_cast<uint64_t*>(address);

    uint64_t old = *address_uint;
    uint64_t assumed; 
    do {
        assumed = old;
        double new_val = *address + val;         
        uint64_t* new_val_uint = reinterpret_cast<uint64_t*>(&new_val);

        old = __sync_val_compare_and_swap(address_uint, assumed, *new_val_uint);        
    }while(assumed != old);    

    return true;
} 

void Sync_batch(int batch_id){
  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }

  int edge_start = h_graph_nodes[batch_start].starting;
  int edge_end = h_graph_nodes[batch_end - 1].starting + h_graph_nodes[batch_end - 1].no_of_edges;

  #pragma omp parallel for  
  for(int i = edge_start; i < edge_end; i++){
    int id = h_graph_edges[i];
    if(mode == 0){
      weight_ind[i] = weight[id];
    }else{
      weight_ind[i] = weight_old[id];
    }
  }
}

void kernel(int batch_id) {
  // Assume mode 0 (SGD)
  double gamma = 0.00005; // the learning rate

  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }
  
  #pragma omp parallel for  
  for(int vid = batch_start; vid < batch_end; vid++){    
    int start = h_graph_nodes[vid].starting;
    int no_of_edges = h_graph_nodes[vid].no_of_edges;                
    int end = start + no_of_edges;              

    double z_vid = 0;  
    for (int i = start; i < end; i++) {
      double xvalue = x_array[i];
      double wvalue;
      if(stale_mode == 0){
        int id = h_graph_edges[i];
        if(mode == 0){
          wvalue = weight[id];
        }else{
          wvalue = weight_old[id];
        }
      }else{
        wvalue = weight_ind[i];
      } 
      z_vid = z_vid + xvalue * wvalue; 
    }
    double y_vid = y_array[vid];
    double sig_gamma = sigmoid(-y_vid * z_vid) * y_vid * gamma;

    for (int i = start; i < end; i++) {
      double xvalue = x_array[i];
      double gradient = sig_gamma * xvalue;

      int id = h_graph_edges[i];
      // Replaced by non-blocking write to original weight
      atomicAdd(&(weight[id]), gradient); 
    }
  }
}