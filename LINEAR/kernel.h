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

#include "node.h"

extern int no_of_nodes;
extern int edge_list_size;

extern Node* h_graph_nodes;
extern int* h_graph_edges;

extern double* y_array;
extern double* x_array;
extern double* weight_old;
extern double* weight;
extern double* weight_ind;

extern int num_threads;
extern int mode;
extern int batch_size;
extern int stale_mode;
extern int stale_threshold;

bool atomicAdd(double* address, double val);
void Sync_batch(int batch_id);
void kernel(int batch_id);
