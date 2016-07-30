#include <pthread.h>
#include "node.h"

void Sync_batch(Node* g_graph_nodes, int* g_graph_edges, int* g_cost, int* g_cost_ind, int no_of_nodes, int batch_size, int batch_id);

void Kernel(Node* g_graph_nodes, int* g_graph_edges, int* g_cost, int* g_cost_ind, bool* g_over, 
	int no_of_nodes, int edge_list_size, int source, int batch_size, int batch_id, int stale_mode);
