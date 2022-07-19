############################
SGL: Scalable Graph Learning
############################

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on 
extremely large datasets. SGL allows users to easily implement scalable graph neural networks and evaluate its
performance on various downstream tasks like node classification, node clustering, and link prediction. Further, SGL
supports auto neural architecture search functionality based
on `OpenBox <https://github.com/PKU-DAIR/open-box>`__. SGL is designed and
developed by the graph learning team from
the `DAIR Lab <https://cuibinpku.github.io/index.html>`__ at Peking University.

----------------------
Library Highlights
----------------------

+ **High scalability**: Follow the scalable design paradigm **SGAP**
  in `PaSca <https://arxiv.org/abs/2203.00638>`__, SGL scale to graph data with
  billions of nodes and edges.
+ **Auto neural architecture search**: Automatically choose decent neural architectures according to specific tasks, and
  pre-defined objectives (e.g., inference time).
+ **Ease of use**: User-friendly interfaces of implementing existing scalable GNNs and executing various downstream
  tasks.


------------------------------------------------
Related Publications
------------------------------------------------

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm** [`PDF <https://arxiv.org/pdf/2203.00638>`__] 
Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui.
The world wide web conference. (WWW 2022, CCF-A)

**Node Dependent Local Smoothing for Scalable Graph Learning** [`PDF <https://arxiv.org/pdf/2110.14377>`__]
Wentao Zhang, Mingyu Yang, Zeang Sheng, Yang Li, Wen Ouyang, Yangyu Tao, Zhi Yang, Bin Cui.
Thirty-fifth Conference on Neural Information Processing Systems. (NeurIPS 2021, CCF-A, Spotlight Presentation, Acceptance Rate: < 3%). 

**Graph Attention Multi-Layer Perceptron** [`PDF <https://arxiv.org/pdf/2108.10097>`__]
Wentao Zhang, Ziqi Yin, Zeang Sheng, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui.
arXiv:2108.10097, 2021. (arXiv preprint). 


------------------------------------------------
License
------------------------------------------------

The entire codebase is under `MIT
license <https://github.com/PKU-DAIR/SGL/blob/main/LICENSE>`__.


.. toctree::
  :caption: Get Started
  :maxdepth: 2
  :titlesonly:

  Overview <get_started/overview/overview>
  Installation <get_started/installation/installation>
  Quick Start <get_started/quick_start/quick_start>


.. toctree:: 
  :caption: API Reference
  :maxdepth: 2

  data <api/data/data>
  datasets <api/datasets/datasets>
  graph operators <api/operators/graph_operators>
  message operators <api/operators/message_operators>
  models <api/models/models>
  tasks <api/tasks/tasks>
  search <api/search/search>
