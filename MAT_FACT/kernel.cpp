#include "kernel.h"

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

  #pragma omp parallel for  
  for(int i = batch_start; i < batch_end; i++){
    int column_index = x_col_array[i];
    for (int k = 0; k < K_count; ++k) {
      R_table_ind[(i * K_count) + k] = R_table[(column_index * K_count) + k];
    }
  }
}

void kernel(int iter, int batch_id){
  // Assume mode 0 (SGD)
  double init_step_size = 0.5;
  double step_size = init_step_size * pow(100.0 + (iter - 1), -0.5);

  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }
  
  #pragma omp parallel for  
  for(int vid = batch_start; vid < batch_end; vid++){    
    // Stack Variables
    double Li_curr[K_count]; double Rj_curr[K_count];
    double Li_update[K_count]; double Rj_update[K_count]; 
    
    int row_index = x_row_array[vid];
    int column_index = x_col_array[vid];
    double Xij = x_val_array[vid];

    for (int k = 0; k < K_count; ++k) {
      Li_curr[k] = L_table[(row_index * K_count) + k];
      if(stale_mode == 0){
        Rj_curr[k] = R_table[(column_index * K_count) + k];
      }else{
        Rj_curr[k] = R_table_ind[(vid * K_count) + k];
      }
    }
    
    double LiRj = 0.0;
    for (int k = 0; k < K_count; ++k) {
      LiRj = LiRj + (Li_curr[k] * Rj_curr[k]);
    }
    
    for (int k = 0; k < K_count; ++k) {
      double gradient = 0.0; 
      double Li_value = Li_curr[k]; 
      double Rj_value = Rj_curr[k];

      gradient = (-2 * Xij * Rj_value) + (2 * LiRj * Rj_value);
      Li_update[k] = -gradient * step_size;
      gradient = (-2 * Xij * Li_value) + (2 * LiRj * Li_value);
      Rj_update[k] = -gradient * step_size;
    }

    // Commit updates
    for (int k = 0; k < K_count; ++k) {
      atomicAdd(&(L_table[(row_index * K_count) + k]), Li_update[k]);
      // Replaced by non-blocking write to original weight
      atomicAdd(&(R_table[(column_index * K_count) + k]), Rj_update[k]); 
    }

    // The loss function at X(i,j) is ( X(i,j) - L(i,:)*R(:,j) )^2.
    atomicAdd(&loss_sum, (double)pow(Xij - LiRj, 2));
  }
}