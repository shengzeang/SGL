#######################
Overview
#######################

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on extremely large datasets. 
SGL allows users to easily implement scalable graph neural networks and evaluate its performance on various downstream tasks like node classification, node clustering, and link prediction. 
Further, SGL supports auto neural architecture search functionality based on `OpenBox <https://github.com/PKU-DAIR/open-box>`__. 
SGL is designed and developed by the graph learning team from the `DAIR Lab <https://cuibinpku.github.io/index.html>`__ at Peking University.



Main Functionalities
------------------------

+ A handy platform for **implementing and evaluating scalable GNNs**.
+ Scalable learning on various graph-related tasks, including **node classification**, **node clustering**, and **link prediction**.
+ **Auto neural architecture search** on given tasks, datasets and objectives.



Training paradigm
-------------------------

The main design goal of SGL is to support scalable graph learning. 
SGL adopts the scalable training paradigm, **SGAP** (Scalable Graph Architecture Paradigm), in `PaSca <https://arxiv.org/abs/2203.00638>`__. 
**SGAP** split the conventional GNN training process into three independent stages --- **Preprocessing**, **Training**, and **Postprocessing**, which can be represented as follows: 

+ **Preprocessing**: :math:`\textbf{M}=graph\_propagate(\textbf{A}, \textbf{X})`; :math:`\textbf{X}'=message\_aggregate(\textbf{M})`
    + **SGAP** propagates and aggregates information at the graph level.

+ **Training**: :math:`\textbf{Y}=model\_train(\textbf{X}')`
    + **SGAP** feeds the propagated and aggregated information into a machine learning model (e.g., SVM, MLP) for training.

+ **Postprocessing**: :math:`\textbf{M}'=graph\_propagate(\textbf{A},\textbf{Y})`; :math:`\textbf{Y}'=message\_aggregate(\textbf{M}')`
    + **SGAP** again propagates and aggregates the outputs of the previous stage at the graph level.


.. note:: 

    The first :math:`message\_aggregate` operation in the **Preprocessing** stage will be transferred to the **Training**  stage if it contains learnable parameters; and the second :math:`message\_aggregate` operation in the **Postprocessing** stage is prohibited to contain learnable parameters.

Compared to conventional GNN training process, **SGAP** has mainly two advantages:

1. The time- and resource-consuming propagation operation is only executed two times during the full training process; while the number of executing propagation in the conventional GNN training process equals to the number of training epochs, which is usually far greater than two.
2. The dependencies between training examples have been fully taken care of in the **Preprocessing** stage. Thus, the training examples can be freely split to small batches to feed into the model in the **Training** stage, which boosts the efficiency and the scalability of the training process.



Model construction paradigm
-------------------------------

Corresponding to its training paradigm, **SGAP**, SGL needs to define the behaviors of two :math:`graph\_propagate`s, two :math:`message\_aggregate`s, and :math:`model\_train` for each GNN model. 
To fulfill this goal, SGL designs three important modules:

+ **Graph Operator**: to carry out the functionality of :math:`graph\_propagate`. It receives the adjacency matrix :math:`\textbf{A}` and the node representation matrix :math:`\textbf{X}`, and outputs a list of propagated information matrices of different propagation depths.
+ **Message Operator**: to carry out the functionality of :math:`message\_aggregate`. It receives a list of propagated information matrices and aggregates the matrices according to pre-defined behaviors. The final output of each **Message Operator** is a single matrix.
+ **Base Model**: to carry out the functionality of :math:`model\_training`. It can be not only deep learning models like MLP, but also traditional machine learning methods like SVM and random forest.

To construct a GNN model in SGL, the users only need to fill in some blanks with pre-/user-defined **Graph Operators**, **Message Operators** and **Base Models**. 
Please refer to `models part <../../api/models/models.html>`__ for the detailed API for constructing models. 
SGL also provides simple interfaces for defining new **Graph Operators** and **Message Operators**, please refer to `operators part <../../api/operators/operators.html>`__ for more details.
