# Graph Learning and Analysis

## Graph
图（graph）是由节点（node）和边（edge）组成的数据结构，用于表示实体及其关系。图在许多领域中都有广泛的应用，例如社交网络、推荐系统、生物信息学等。

## Graph Learning
图学习（graph learning）是指利用机器学习方法来分析和处理图数据的技术。它包括图嵌入（graph embedding）、图神经网络（graph neural networks, GNNs）等方法，旨在从图结构中提取有用的信息，以进行节点分类、边预测、图分类等任务。

## Project 1: Privacy in Graph
负责人：徐闽泽，谢祯泰，王智彬

图数据通常包含敏感信息，如个人隐私、商业机密等。一种常用的构图方式是把用户当做节点，用户之间的关系（如好友关系、关注关系等）当做边。因此用户之间的边往往包含敏感信息。如何在保护隐私的前提下进行图分析是一个重要的研究课题。

我们已经发表相关论文[Sectric: Towards Accurate, Privacy-preserving and Efficient Triangle Counting](https://www.vldb.org/pvldb/vol18/p3382-xu.pdf)

下一步希望扩展到更复杂的分析及GNNs上。


## Project 2: Expressive Power of GNNs
负责人：王智彬

图神经网络（GNNs）是一类专门用于处理图数据的神经网络模型。GNNs通过聚合节点及其邻居的信息来学习节点的表示。GNNs的表达能力是指其能够区分不同图结构的能力。研究GNNs的表达能力有助于理解其局限性，并指导新模型的设计。

项目目标，参与我们的GNNs表达能力的研究，即扩展下述[论文](https://arxiv.org/pdf/2505.19188)