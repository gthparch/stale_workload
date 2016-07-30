#ifndef _CPP_DATA_LOADER_
#define _CPP_DATA_LOADER_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include "node.h"

using namespace std;


template < class T> 
void convert_from_string(T& value, const string& s){
    stringstream ss(s);
    ss >> value;
}

class SimpleDataLoader {
    private:
        int get_cat(const string& data) {
            int c;
            convert_from_string(c, data);

            return c;
        }

        bool get_features(const string& data, int& index, double& value) {
            int pos = data.find(":");
            if (pos == -1) return false;
            convert_from_string(index, data.substr(0, pos));
            convert_from_string(value, data.substr(pos + 1));

            return true;
        }

        // please note we need to add a default feature to each instance and set the feature weight to 1
        bool get_edges(const string& line, const int line_num, Node* h_graph_nodes) {
            if (line.empty()) {
                return false;
            }

            size_t start_pos = 0;
            char space = ' ';
            
            // the dummy feature 
            h_graph_nodes[line_num].no_of_edges++; 

            while (true) {
                size_t pos = line.find(space, start_pos);
                string data = line.substr(start_pos, pos - start_pos);
                if (!data.empty()) {
                    if (start_pos != 0) {
                        int index = -1;
                        double v = 0;
                        get_features(data, index, v);
                        if (index != -1) {
                            h_graph_nodes[line_num].no_of_edges++; 
                        }
                    }
                }
                if ((int)pos != -1) {
                    start_pos = pos + 1;
                } else {
                    break;
                }
            }
            return true;
        }        

        // please note we need to add a default feature to each instance and set the feature weight to 1
        bool parse_line(const string& line, int& cat, const int line_num, 
            double* x_array, Node* h_graph_nodes, int* h_graph_edge) {            
            if (line.empty()) {
                return false;
            }

            int edge_index = h_graph_nodes[line_num].starting;

            size_t start_pos = 0;
            char space = ' ';
            
            // the dummy feature 
            x_array[edge_index] = 1;
            h_graph_edge[edge_index] = 0;
            edge_index++;

            while (true) {
                size_t pos = line.find(space, start_pos);
                string data = line.substr(start_pos, pos - start_pos);
                if (!data.empty()) {
                    if (start_pos == 0) {
                        cat = get_cat(data);
                    } else {
                        int index = -1;
                        double v = 0;
                        get_features(data, index, v);
                        if (index != -1) {
                            x_array[edge_index] = v;
                            h_graph_edge[edge_index] = index;
                            edge_index++;
                        }
                    }
                }
                if ((int)pos != -1) {
                    start_pos = pos + 1;
                } else {
                    break;
                }
            }
            return true;
        }


    public:
        SimpleDataLoader(){}

       void setup_nodes(const char* file_path, Node* h_graph_nodes) {
            ifstream in(file_path);
            string line;
            int line_num = 0;
            if (in.is_open()) {
                while (in.good()) {
                    getline(in, line);
                    if (line.empty()) {
                        continue;
                    }
                    if (!get_edges(line, line_num, h_graph_nodes)) {
                        cout << "get_edges: " << line << ", failed.." << endl;
                        continue;
                    }
                    line_num += 1;
                }
                in.close();
            }
        }        

        void load_file(const char* file_path, double* y_array, double* x_array, 
            Node* h_graph_nodes, int* h_graph_edges) {            
            ifstream in(file_path);
            string line;
            int line_num = 0;
            if (in.is_open()) {
                while (in.good()) {
                    getline(in, line);
                    if (line.empty()) {
                        continue;
                    }
                    int cat = 0;
                    if (!parse_line(line, cat, line_num, x_array, h_graph_nodes, h_graph_edges)) {
                        cout << "parse line: " << line << ", failed.." << endl;
                        continue;
                    }
                    y_array[line_num] = cat;
                    line_num += 1;
                }
                in.close();
            }
        }
};

#endif
