#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include <time.h>

#include "node.h"
#include "kernel.h"

using namespace std;

#define MAX_COST 1000000000

FILE *fp;
int source;
int num_threads;
int batch_size;
int stale_mode;
int stale_threshold;

int no_of_nodes;
int edge_list_size;
Node* h_graph_nodes;
int* h_graph_edges;
int* h_node_color;
Node* h_ind_nodes;
int* h_ind_edges;
int* h_node_color_ind;
bool g_stop;

void compute(int source){
  int iteration = 0;
  int num_batches = no_of_nodes / batch_size;
  if(num_batches * batch_size < no_of_nodes){
    num_batches++;
  }

  while (true){
    g_stop = false;

    for(int batch_id = 0; batch_id < num_batches; batch_id++){
      Kernel(h_graph_nodes, h_graph_edges, h_node_color, h_node_color_ind, &g_stop, no_of_nodes, edge_list_size, batch_size, batch_id, stale_mode);

      if(stale_mode == 1){
        int next_batch_id = batch_id + stale_threshold;
        if(next_batch_id >= num_batches){
          next_batch_id = next_batch_id % num_batches;
        }
        Sync_batch(h_graph_nodes, h_graph_edges, h_node_color, h_node_color_ind, no_of_nodes, batch_size, next_batch_id);  
      }
    }
    
    iteration++;
    if(g_stop == false){
      break;
    }
  }  
  cout << "iteration: " << iteration << endl;   
}

int main( int argc, char** argv) {
	no_of_nodes=0;
	edge_list_size=0;

  h_graph_nodes = NULL;
  h_graph_edges = NULL;
  h_node_color = NULL;    
  
  fp = fopen(argv[1], "r");
  num_threads = atoi(argv[2]);
  batch_size = atoi(argv[3]);
  stale_mode = atoi(argv[4]);
  stale_threshold = atoi(argv[5]);
  int dummy = fscanf(fp,"%d",&no_of_nodes);

  // allocate host memory    
  h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
  h_node_color = (int*) malloc( sizeof(int)*no_of_nodes);        

  int start, edgeno, node_rand_value;
  // initalize the memory
  for( unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp,"%d %d %d",&start,&edgeno,&node_rand_value);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_node_color[i]=0;
  }

  //read the source node from the file
  dummy = fscanf(fp,"%d",&source);

  //read the destination node from the file
  int destination;    
  fscanf(fp,"%d",&destination);

  dummy = fscanf(fp,"%d",&edge_list_size);

  h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);

  int id,cost;
  for(int i=0; i < edge_list_size ; i++) {
    dummy = fscanf(fp,"%d",&id);
    dummy = fscanf(fp,"%d",&cost);
    h_graph_edges[i] = id;
  }

  if(fp)
    fclose(fp);    

  h_node_color[source] = 1;   

  h_node_color_ind = (int*) malloc( sizeof(int)*edge_list_size);    
  for( unsigned int i = 0; i < edge_list_size; i++) {
    int id = h_graph_edges[i];
    h_node_color_ind[i]= h_node_color[id];
  }

  compute(source);

	//Store the result into a file
  string result_file("result.txt." + to_string(batch_size) + "." + to_string(stale_mode) + "." + to_string(stale_threshold));
  FILE *fpo = fopen(result_file.c_str(), "w");
	for(int i=0;i<no_of_nodes;i++){
    fprintf(fpo,"%d) color:%d\n",i,h_node_color[i]);
  }
	fclose(fpo);

	// cleanup memory
	free(h_graph_nodes);
	free(h_graph_edges);
  free(h_node_color);
}
