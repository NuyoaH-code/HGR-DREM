#Document-Level Relation Extraction Based on Heterogeneous Graph Reasoning
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
>The purpose of document-level relation extraction is to mine semantic information from multiple sentences in a document and identify the relations between entities across sentences. However, it is challenging to fully express the information contained in documents and reason about cross-sentence entities with reasonable and efficient reasoning mechanisms. To overcome this issue, we propose the Document-Level Relation Extraction Model based on Heterogeneous Graph Reasoning (HGR-DREM), which efficiently extracts relation information in documents by using document-level heterogeneous graphs and reasoning strategies based on the heterogeneous graphs. Specifically, we first construct a document-level heterogeneous graph, which can fully express the semantic relations between entities in the document. Second, a reasoning mechanism based on the meta-path attention is designed to strengthen the mutual influence of nodes in the meta-path. We also designed an extended adjacency matrix to represent the heterogeneous graph and used the graph convolutional neural networks (GCNs) to extract high-dimensional features. We conduct experimental studies based on three real-world datasets. Compared with the traditional methods, our method achieved the best results and demonstrated the efficiency in the task of document-level relation extraction.
# Environments<br>
* Ubuntu-18.10.1(4.18.0-25-generic)<br>
* Python(3.6.8)<br>
* Cuda(10.1.243)<br>
# Dependencies<br>
* matplotlib (3.3.2)<br>
* networkx (2.4)<br>
* nltk (3.4.5)<br>
* numpy (1.19.2)<br>
* torch (1.3.0)<br>

# Data<br>
First you should get pretrained Bert_base model from [huggingface](https://github.com/huggingface/transformers) and put it into `./bert/bert-base-uncased/`. <br>
Before running our code you need to obtain the DocRED dataset from the author of the dataset, [Here](https://github.com/thunlp/DocRED).<br>
After downing DocRED, you can use `gen_data_extend_graph.py` to preprocess data for Glove-HDR-DREM and use `gen_bert_data_extend_graph.py` to preprocess data for BERT-HDR-DREM. Finally, processed data will be saved into `./prepro_data` and `./prepro_data_bert` respectively.<br> 
For the CDR, you can obtain it from https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/.
For the GDA, you can obtain it from https://bitbucket.org/alexwuhkucs/gda-extraction/src/master/.
# Run code<br>
`train.py` used to start training<br>
`test.py` used to evaluation model's performance on Dev or Test set.<br>
`Config.py` is for training Glove-based model And `Config_bert.py` is used for training Bert_based model
# Evaluation<br>
For Dev set, you can use `test.py` to evaluate you trained model.
For Test set, you should first use `test.py` to get test results which saved in `./result`, and submit it into [Condalab competition](https://competitions.codalab.org/competitions/20717).