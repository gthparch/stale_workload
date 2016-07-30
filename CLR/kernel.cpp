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

void Sync_batch(Node* g_graph_nodes, int* g_graph_edges, int* g_node_color, int* g_node_color_ind, int no_of_nodes, int batch_size, int batch_id){
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
    g_node_color_ind[i] = g_node_color[id];
  }
}

void Kernel(Node* g_graph_nodes, int* g_graph_edges, int* g_node_color, int* g_node_color_ind, bool *g_over, 
  int no_of_nodes, int edge_list_size, 
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

    bool violated = false;
    int my_color = g_node_color[vid];

    for (int i = start; i < end; i++) {
      int id = g_graph_edges[i];
      if (vid < id) {
        int neighbor_color;
        if(stale_mode == 0){
          neighbor_color = g_node_color[id];        
        }else{
          neighbor_color = g_node_color_ind[i];        
        }
        if (my_color >= neighbor_color) {
          violated = true;
          my_color = neighbor_color;
        }
      }
    }
    if (violated == true) {        
      my_color = my_color - 1;
      g_node_color[vid] = my_color; 
      *g_over = true;
    }        
  }
}

#endif //_KERNEL_H_
