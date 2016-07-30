#ifndef _CPP_DATA_LOADER_
#define _CPP_DATA_LOADER_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace std;


template < class T> 
void convert_from_string(T& value, const string& s){
    stringstream ss(s);
    ss >> value;
}

class SimpleDataLoader {
    private:
        vector<int> X_row; 
        vector<int> X_col; 
        vector<double> X_val; 

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
        bool parse_line(const string& line, int& cat, const int line_num, 
            vector<int>& X_row, vector<int>& X_col, vector<double>& X_val) {            
            if (line.empty()) {
                return false;
            }

            size_t start_pos = 0;
            char space = ' ';
            
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
                            X_row.push_back(line_num);
                            X_col.push_back(index - 1);
                            X_val.push_back(v);
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

        int load_file(const char* file_path, int*& x_row_array, int*& x_col_array, double*& x_val_array) {            
            X_row.clear();
            X_col.clear();
            X_val.clear();            

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
                    if (!parse_line(line, cat, line_num, X_row, X_col, X_val)) {
                        cout << "parse line: " << line << ", failed.." << endl;
                        continue;
                    }
                    line_num += 1;
                }
                in.close();
            }

            int no_of_nodes = X_row.size();

            x_row_array = (int*) calloc(no_of_nodes, sizeof(int));      
            x_col_array = (int*) calloc(no_of_nodes, sizeof(int));      
            x_val_array = (double*) calloc(no_of_nodes, sizeof(double));      

            for(int i = 0; i < no_of_nodes; i++){
                x_row_array[i] = X_row[i];
                x_col_array[i] = X_col[i];
                x_val_array[i] = X_val[i];
            }

            return no_of_nodes;
        }
};


#endif
