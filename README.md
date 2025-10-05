# Structural Entropy Guided Meta Learning for Few-shot Node Classification

To run the proposed model in the paper, SE-FSNC and SE-FSGC, see the corresponding folder:

* Few-Shot Node Classification: [code]
  
  The specific parameter settings of the model can be found in params.py in the code directory. The entry point of the main program is main.py. If you want to run the 10-way 1-shot node classification task on the CoraFull dataset, you can execute the following command:
  ```
  cd code
  ```
  ```
  python main.py --dataset_name "ogbn-arxiv" --n_way 10 --k_shot 1 --runs 10
  ```
  You can run few-shot tasks on other datasets by modifying the "dataset_name" parameter, as well as the "n_way" and "k_shot" parameters.

* Few-Shot Graph Classification: [SE-FSGC]

  If you want to perform a few-shot graph classification task, you can execute the following command:
  ```
  cd SE-FSGC
  ```
  First, unzip the dataset.
  ```
  unzip data.zip
  ```

  Then, transform the graph to its corresponding coding tree.
  ```
  python trans_graph.py
  ```

  Finally, run the few-shot graph classification task.

  ```
  python train_reddit.py or python train_enzymes.py
  ```
