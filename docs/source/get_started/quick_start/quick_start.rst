###################
Quick Start
###################

In this short tutorial, we will qucikly go through the basic and the advanced usage of SGL. 
The tutorial is composed of following parts:

.. contents::
    :local:


Basic usage
______________________________________________

In this part, we will introduce the basic usage of SGL, including how to excute graph-related tasks and how to use the NAS (Neural Architecture Search) functionality.

___________________________________
Execute graph-related tasks
___________________________________

SGL provides user-friendly interfaces to execute graph-related tasks, including node classification, node clustering, and link prediction.
In this tutorial, we will go through an example of excuting a node classification task with a SGC on the PubMed dataset.

Import datasets
>>>>>>>>>>>>>>>>>>>>>>

Here we import the PubMed dataset via the following code:

.. code:: python

    from sgl.datasets import Planetoid
    dataset = Planetoid(name="pubmed", root="./", split="official")

The :obj:`Planetoid` class contains three popular graph datasets: Cora, Citeseer, and PubMed. 

+ The 1st argument :obj:`name` indicates which dataset among the three to choose; 
+ The 2nd argument :obj:`root` indicates where to put the dataset files;
+ The 3rd argumnet :obj:`split` indicates the train/validation/test split.

SGL has integated many graph datasets other than the Planetoid datasets.
Please refer to `datasets part <../../api/datasets/datasets.html>`__ for the detailed information of each dataset.


Import GNN models
>>>>>>>>>>>>>>>>>>>>>>>

Here we import the GNN model SGC as follows:

.. code:: python

    from sgl.models.homo import SGC
    model = SGC(prop_steps=3, feat_dim=dataset.num_features, output_dim=dataset.num_classes)

SGL supports not only GNNs designed for homogeneous graphs but also GNNs designed for heterogeneous graphs.
The two different categories of models reside in :obj:`sgl.models.homo` and :obj:`sgl.models.hetero`, respectively.
The GNN model SGC is designed for homogeneous graphs, and thus can be imported from :obj:`sgl.models.homo`.
:obj:`SGC` class has three main arguments:

+ The 1st argument :obj:`prop_steps` stands for the propagation depth;
+ The 2nd argument :obj:`feat_dim` stands for the dimension of the input feature;
+ The 3rd argument :obj:`output_dim` stands for the dimension of the output representation.

Please refer to the `models part <../../api/models/models.html>`__ for more details of SGC and other GNN models.

   
Execute tasks
>>>>>>>>>>>>>>>>>>>>>>>>
The node classification task can be executed by the following code:

.. code:: python

    from sgl.tasks import NodeClassification
    device = "cuda:0"
    test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-5, epochs=200, device=device).test_acc

The users have to input the adopted dataset, the adopted GNN model, and several hyperparameters before executing a task.

The possible output of the above code might be:

.. code:: bash

    Preprocessing done in 0.1280s
    Epoch: 001 loss_train: 1.0985 acc_train: 0.3333 acc_val: 0.2300 acc_test: 0.2110 time: 1.3086s
    Epoch: 002 loss_train: 1.0289 acc_train: 0.3667 acc_val: 0.7100 acc_test: 0.6920 time: 0.0030s
    Epoch: 003 loss_train: 0.9554 acc_train: 0.8667 acc_val: 0.7220 acc_test: 0.7300 time: 0.0040s
    Epoch: 004 loss_train: 0.8918 acc_train: 0.9333 acc_val: 0.7220 acc_test: 0.7300 time: 0.0030s
    Epoch: 005 loss_train: 0.8354 acc_train: 0.9167 acc_val: 0.7400 acc_test: 0.7220 time: 0.0020s
    Epoch: 006 loss_train: 0.7835 acc_train: 0.9333 acc_val: 0.7380 acc_test: 0.7180 time: 0.0030s
    Epoch: 007 loss_train: 0.7358 acc_train: 0.9167 acc_val: 0.7280 acc_test: 0.7240 time: 0.0020s
    Epoch: 008 loss_train: 0.6929 acc_train: 0.9333 acc_val: 0.7320 acc_test: 0.7320 time: 0.0030s
    Epoch: 009 loss_train: 0.6546 acc_train: 0.9333 acc_val: 0.7360 acc_test: 0.7340 time: 0.0030s
    Epoch: 010 loss_train: 0.6198 acc_train: 0.9333 acc_val: 0.7360 acc_test: 0.7360 time: 0.0030s
    ......
    Epoch: 191 loss_train: 0.1886 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 192 loss_train: 0.1886 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7930 time: 0.0030s
    Epoch: 193 loss_train: 0.1885 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 194 loss_train: 0.1884 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 195 loss_train: 0.1884 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 196 loss_train: 0.1883 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0020s
    Epoch: 197 loss_train: 0.1882 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7930 time: 0.0040s
    Epoch: 198 loss_train: 0.1882 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 199 loss_train: 0.1881 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7920 time: 0.0030s
    Epoch: 200 loss_train: 0.1880 acc_train: 1.0000 acc_val: 0.8020 acc_test: 0.7910 time: 0.0030s
    Optimization Finished!
    Total time elapsed: 1.9751s
    Best val: 0.8020, best test: 0.7920

Please refer to the `tasks part <../../api/tasks/tasks.html>`__ for more details of executing graph-related tasks.

_________________________________________
Auto neural architrcture search (TODO)
_________________________________________




Advanced usage
___________________________________

In this part, we will introduce the advanced usage of SGL, including adopting user-defined datasets, building models under SGAP paradigm, implementing new graph operators and message operators.


_______________________________________
Adopt user-defined datasets
_______________________________________

SGL designs two base classes, :obj:`NodeDataset` and :obj:`HeteroNodeDataset`, for the homogeneous graph datasets and the heterogeneous graph datasets, respectively.
We will take implementing a homogeneous graph dataset as an example below to explain how to adopt user-defined datasets.

To implement a new homogeneous graph dataset, one has to first to inherit the base class :obj:`NodeDataset`, whose detailed introduction can be found in the `data part <../../api/data/data.html>`__.
Then, there exist two important virtual functions to implement: 

+ :obj:`download`: download the raw files of the dataset from the Interent and store them in pre-defined places;
+ :obj:`process`: process the raw files fetched by :obj:`download` and store the processed file defined by the data class :obj:`Graph`.

The data class :obj:`Graph` is designed to store the critical data for the homogeneous graph; the corresponding data class for the heterogeneous graph is :obj:`HeteroGraph`.
To instantiate :obj:`Graph`, one needs to at least provide the following information:

+ :obj:`row`: the row index of the edges in the graph;
+ :obj:`col`: the column index of the edges in the graph; 
+ :obj:`edge_weight`: the weight of the edges in the graph;
+ :obj:`edge_type`: the type of the edges in the graph;
+ :obj:`num_node`: the total number of nodes in the graph;
+ :obj:`node_type`: the type of the nodes in the graph.

The datasets in the `datasets part <../../api/datasets/datasets.html>`__ all follow the same construction scheme.

Please refer to the `data part <../../api/data/data.html>`__ for more detailed introduction of the two base classes, :obj:`NodeDataset` and :obj:`HeteroNodeDataset`.


____________________________________________
Build models under SGAP paradigm
____________________________________________

SGL adopts the `SGAP <https://arxiv.org/abs/2203.00638>`__ (Scalable Graph Architecture Paradigm) as its training paradigm.
Corresponding to that, the model construction paradigm differs from the conventional `message passing <http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf>`__ paradigm.
The detailed introduciton of the model construction paradigm of SGL is provided in `overview <../overview/overview.html>`__.
Below will explain how to build a SGC in SGL.

As introduced in `overview <../overview/overview.html>`__, a GNN model in SGL is composed of five parts:

+ *pre_graph_op*, *pre_msg_op*: **Graph Operator** and **Message Operator** for the Preprocessing stage;
+ *base_model*: **Base Model** for the Training stage;
+ *post_graph_op*, *post_msg_op*: **Graph Operator** and **Message Operator** for the Postprocessing stage.

Thus, users only have to assign each module with pre-/user-defined Graph Operator/Message operator/Base Model when building models after inheriting the base class :obj:`BaseSGAPModel`.
The behaviors of the adopted different Graph Operators, Message Operators and Base Models determine the behaviors of the built GNN models.
The code of building SGC is provided below:

.. code:: python

    from sgl.models.base_model import BaseSGAPModel
    from sgl.models.simple_models import LogisticRegression
    from sgl.operators.graph_op import LaplacianGraphOp
    from sgl.operators.message_op import LastMessageOp


    class SGC(BaseSGAPModel):
        def __init__(self, prop_steps, feat_dim, output_dim):
            super(SGC, self).__init__(prop_steps, feat_dim, output_dim)

            self._pre_graph_op = LaplacianGraphOp(prop_steps, r=0.5)
            self._pre_msg_op = LastMessageOp()
            self._base_model = LogisticRegression(feat_dim, output_dim)

.. note:: 

    The *LaplacianGraphOp*, *LastMessageOp*,and *LogisticRegreesion* are pre-defined Graph Operator, Message Operator, and Base Model, respectively. 

.. note:: 

    SGC does not have the Postprocessing stage in its training process. Thus, the modules used for the Postprocessing stage do not exist in the construction of SGC.

In the following parts of this tutorial, we will introduce ways to implement new Graph Operators and Message Operators.


________________________________________
Implement new Graph Operators
________________________________________

As introduced in `overview <../overview/overview.html>`__, the behaviors of the Graph Operators can be represented as follows: :math:`\textbf{M}=graph\_propagate(\textbf{A}, \textbf{X})`.
Thus, the critical part of implementing new Graph Operators is to determine the value of the matrix :math:`\textbf{A}`.

In SGL, users only need to implement the virtual function *construct_adj*, which takes in the original adjacency matrix of the graph and outputs the desired propagation matrix after inheriting the base class :obj:`GraphOp`.
Below is the implementation of the PPR (Personalized PageRank) Graph Operator:

.. code:: python

    class PprGraphOp(GraphOp):
        def __init__(self, prop_steps, r=0.5, alpha=0.15):
            super(PprGraphOp, self).__init__(prop_steps)
            self.__r = r
            self.__alpha = alpha

        def _construct_adj(self, adj):
            adj_normalized = adj_to_symmetric_norm(adj, self.__r)
            adj_normalized = (1 - self.__alpha) * adj_normalized + self.__alpha * sp.eye(adj.shape[0])
            return adj_normalized.tocsr()

Please refer to `operators part <../../api/operators/operators.html>`__ for more detailed introduction.


_________________________________________
Implement new Message Operators
_________________________________________

Similar to implementing new Graph Operators, implementing new Message Operators is easy in SGL.
The users need to determine the behaviors of the new Message Operators represented in :math:`\textbf{X}'=message\_aggregate(\textbf{M})`.

Practically speaking, users have to implement the virtual function *combine* function after inheriting the base class :obj:`MessageOp`.
The code below provides the implementation of the ConcatMessageOp in SGL:

.. code:: python

    class ConcatMessageOp(MessageOp):
        def __init__(self, start, end):
            super(ConcatMessageOp, self).__init__(start, end)
            self._aggr_type = "concat"

        def _combine(self, feat_list):
            return torch.hstack(feat_list[self._start:self._end])

Please refer to `operators part <../../api/operators/operators.html>`__ for more detailed introduction.

