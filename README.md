# stale_workload
Stale value tolerant workload OpenMP version.

### Compile/run 

Make

sh run_test.sh

### Code 
The code is modified from original implmentation

BINARY, LINEAR
- Implemented from scratch

MAT_FACT
- From Petuum (https://github.com/petuum/bosen/tree/v0.9.3/apps/matrixfact)

BFS, SSSP
- From P. Harish and P. J. Narayanan. 
- Accelerating Large Graph Algorithms on the GPU using CUDA in HiPC 07, 2007.
- Code available at https://researchweb.iiit.ac.in/~harishpk/Codes/HiPC.zip

CC, NP
- Implemented based on K. Vora, S. C. Koduru, and R. Gupta. 
- ASPIRE: Exploiting Asynchronous Parallelism in Iterative Algorithms Using a Relaxed Consistency Based DSM in OOPSLA 14, 2014.

CLR 
- Implemented based on N. Lakshminarayana and H. Kim. 
- Spare register aware prefetching for graph algorithms on GPUs in HPCA 14, 2014.

### Dataset 
Available at https://drive.google.com/file/d/0B7larI93dpUgX0dNbHVtdE05NTQ/view?usp=sharing

The inputs are formated into our input format from original inputs. 

coAuthorsDBLP
- From DynoGraph(https://github.com/sirpoovey/DynoGraph)
- Original input available at https://github.com/sirpoovey/DynoGraph/tree/master/data

LBDC-1000k
- From GraphBIG(https://github.com/graphbig/graphBIG)
- Original input available at https://github.com/graphbig/graphBIG/wiki/GraphBIG-Dataset

news20.binary
- From LIBSVM dataset(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
- Original input available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
- Source: 

    [LIBSVM] Chih-Chung Chang and Chih-Jen Lin.
    LIBSVM : a library for support vector machines. 
    ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011

    [SSK05a] S. Sathiya Keerthi and Dennis DeCoste. 
    A modified finite Newton method for fast solution of large scale linear SVMs. 
    Journal of Machine Learning Research, 6:341-361, 2005.
