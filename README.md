# Substructure Enhanced Contexualized K-hop Message-Passing GNNs
This repository contains code and other resources for our work in KDD23': `Improving the Expressiveness of K-hop Message Passing GNNs by Injecting Contextualized Substructure Information`, the architecture of SEK-GNN is illustrated in the figure:

<img src="https://github.com/hsaduasdb23/Expressiveness_K_hop_GNNs/blob/main/sek-gnn.png" width="600" height="420" alt="SEK-GNN Architecture"/><br/>

### Code Dependencies

The code relies on the following libraries: 

1) Pytorch 2) Pytorch-Geometric 3) Numpy 4) Pandas 5) Pytorch-Lightning 6) Scikit-Learn  7) TermColor 8) nvidia-smi 9) schedule 10) datetime 11) argparse 12) glob 13) rdkit (IMPORTANT FOR QM9 dataset, if not installed, the data preprocessing will encounter some problem)

## How to run the experiment

### Graph Property Dataset
1. To run SEK-GIN, please follow these steps:

   i) Preprocessing dataset. In the root directory, run `python data_pna.py`, it will generate the GraphProperty dataset suitable for SEK-GIN, and preprocess it.

   ii) To try our code, use the code from `run_graph_property.py`, for complete arguments, please see the detailed code and comment in run_graph_property.py. For example, to run a 5-layer 6-hop SEK-GIN with geometric combine function and attention-based pooling function for task 0, run the following code:
     
     ```python run_graph_property.py --mhc_layer_num 5 --mhc_num_hops 6 --lr 8e-3 --pooling_method attention --combine geometric --task 0```

2. To run SEK-PPGN, go to `GraphProperty/` directory, and then use the provided code `run_gp_ppgn.py`, e.g., to run 1-layer 3-hop SEK-PPGN for task 1, run the following code:
  
    ```python run_gp_ppgn.py --use_ppgn 1 --hop 3 --task 0``` 
  
  
  ### Subgraph Count Dataset
  1. To run SEK-GIN, please follow these steps:
   
   i) Preprocessing dataset. In the root directory, run `python SubgraphCount.py`, it will generate the GraphCount dataset suitable for SEK-GIN, and preprocess it.
   
   ii) To try our code, use the code `run_subgraph_count.py`, for complete arguments, please see the detailed code and comment in run_subgraph_count.py. For example, to run a 5-layer 6-hop SEK-GIN with geometric combine function and attention-based pooling function for task 1, run the following code(for GraphCount dataset, the pooling method is default to be attention, no choice is provided in the command line):
   
     ```python run_subgraph_count.py --mhc_layer_num 5 --mhc_num_hops 6 --lr 8e-3 --combine geometric --task 0```
  
  2. To run SEK-PPGN, go to `GraphCount/` directory, and then use the provided code `run_graphcount_ppgn.py`, e.g., to run 1-layer 3-hop SEK-PPGN for task 1, run the following code:
    
      ```python run_graphcount_ppgn.py --use_ppgn 1 --hop 3 --task 1``` 
      
  ### TU Dataset
  1. Dataset preprocessing
  
  To preprocess dataset, run `python dataset_processing.py --dataset_name ${dataset_name}` in root directory, e.g.,
  
      ```python dataset_processing.py --dataset_name MUTAG``` 
  
  2. Run 10 folds in parallel
  
  To make the code efficient, our code runs 10 folds in parallel using `run_tu_cmd_fold_parallel.py`. For example, to run experiment on MUTAG dataset using a 2-layer 2-hop SEK-GIN with geometric combine function, run the following command:
  
      ```python run_tu_cmd_fold_parallel.py --dataset_name MUTAG --combine geometric --layer_num 2 --num_hops 2```
  
  To run a hyper-parameter search, run the following command:
  
      ```python run_tu_cmd_fold_parallel.py --dataset_name MUTAG --search ```
      
  The experimental data will be saved in the disk, then run `python show_results.py --which_dataset ${dataset_name}`, this will calculate the final result in both settings described in the paper and save a csv file named `results.csv` in the root directory. 


### QM9 Dataset
1. Preprocess dataset

To download and preprocess the dataset, go to directory `QM9/`, and run `python QM9Dataset.py`.

2. Run the experiment
To run the experiment for a 6-layer 5-hop SEK-GIN with geometric combine function and attention pooling for target 2, run the following code:

      ```python run_QM9.py --base_gnn GIN --mhc_num_layers 6 --mhc_num_hops 5 --combine geometric  --task 2 ```

For a detailed explanation for each argument, see the code and comments in `run_QM9.py`.


### Reproducibility
1. TU dataset

For TU dataset, one can use `--search` argument and run the command `python run_tu_cmd_fold_parallel.py --dataset_name MUTAG --search ` to reproduce the result, as mentioned earlier.

2. Other datasets

For GraphProperty dataset, GraphCount and QM9 dataset, as the search space is not large, it is recommended to use the command provided earlier and TMUX to open multiple windows to try various hyperparameter combinations as suggested in the Experiment Setting section in the paper.


### Experiment Result

#### Synthetic dataset

<img src="https://github.com/hsaduasdb23/Expressiveness_K_hop_GNNs/blob/main/gp_result.png" width="700" height="320" alt="GPGC"/><br/>

#### TU dataset

<img src="https://github.com/hsaduasdb23/Expressiveness_K_hop_GNNs/blob/main/tu_result.png" width="700" height="320" alt="GPGC"/><br/>

#### QM9 dataset

<img src="https://github.com/hsaduasdb23/Expressiveness_K_hop_GNNs/blob/main/qm9_result.png" width="700" height="320" alt="QM9"/><br/>




__References__

Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699, 2020.

Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent L ́etourneau, and Prudencio Tossou. Re-thinking graph transformers with spectral attention. Advances in Neural Information Processing Systems, 34:21618–21629, 2021.

Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in Neural Information Processing Systems, 34:28877–28888, 2021.



