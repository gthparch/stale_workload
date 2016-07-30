#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <pthread.h>
#include <assert.h>
#include <time.h>

#include "node.h"

using namespace std;

#define MAX_COST 1000000000

bool atomicMin(int* address, int val){
    int old = *address;
    int assumed;

    do {
        assumed = old;
        if (val < assumed){
            old = __sync_val_compare_and_swap(address, assumed, val);                                
        }else{
            return false;
        }
    }while(assumed != old);

    return true;
} 

void Sync_batch(Node* g_graph_nodes, int* g_graph_edges, int* g_cost, int* g_cost_ind, int no_of_nodes, int batch_size, int batch_id){
  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }

  int edge_start = g_graph_nodes[batch_start].starting;
  int edge_end = g_graph_nodes[batch_end - 1].starting + g_graph_nodes[batch_end - 1].no_of_edges;

  #pragma omp parallel for  
  for(int i = edge_start; i < edge_end; i++){
    int id = g_graph_edges[i];
    g_cost_ind[i] = g_cost[id];
  }
}


void Kernel_s(Node* g_graph_nodes, int* g_graph_edges, int* g_cost, int* g_cost_ind, bool* g_over, 
  int no_of_nodes, int edge_list_size, int source, 
  int batch_size, int batch_id, int stale_mode){
  
  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }

  #pragma omp parallel for  
  for(int vid = batch_start; vid < batch_end; vid++){    
    int start = g_graph_nodes[vid].starting;
    int no_of_edges = g_graph_nodes[vid].no_of_edges;                
    int end = start + no_of_edges;  

    int min_cost = MAX_COST; 
    for(int i = start; i < end; i++){
      int id, neighbor_cost;
      if(stale_mode == 0){
        id = g_graph_edges[i];
        neighbor_cost = g_cost[id];
      }else{
        neighbor_cost = g_cost_ind[i];
      }
      
      if(min_cost > neighbor_cost){
        min_cost = neighbor_cost;
      }
    }
    int my_cost = g_cost[vid]; 
    int new_cost = min_cost + 1;     
    if(my_cost > new_cost){
      g_cost[vid] = new_cost; 
      *g_over = true; 
    }
  }
}

void Kernel_m(Node* g_graph_nodes, int* g_graph_edges, int* g_cost, int* g_cost_ind, bool* g_over, 
  int no_of_nodes, int edge_list_size, int source, 
  int batch_size, int batch_id, int stale_mode){
  
  int batch_start = batch_id * batch_size;
  int batch_end = batch_start + batch_size;
  if(batch_end > no_of_nodes){
    batch_end = no_of_nodes;
  }

  #pragma omp parallel for  
  for(int vid = batch_start; vid < batch_end; vid++){  
    int start = g_graph_nodes[vid].starting;
    int no_of_edges = g_graph_nodes[vid].no_of_edges;                
    int end = start + no_of_edges;  

    int my_cost = g_cost[vid]; 
    int new_cost = my_cost + 1;           
    for(int i = start; i < end; i++){
      int id = g_graph_edges[i];
      if(stale_mode == 0){
        if(atomicMin(g_cost + id, new_cost)){
          *g_over = true; 
        }
      }else{
        int neighbor_cost = g_cost_ind[i];
        if(neighbor_cost > new_cost){
          // Replaced by non-blocking write to original weight
          atomicMin(g_cost + id, new_cost);
          *g_over = true; 
        }
      }
    }
  }
}

#endif //_KERNEL_H_

