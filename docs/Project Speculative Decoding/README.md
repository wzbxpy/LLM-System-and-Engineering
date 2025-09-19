# 大模型推理优化-投机解码性能优化

## 项目背景

随着大规模语言模型（LLM）在各类自然语言处理任务中的广泛应用，模型推理效率成为制约其落地和推广的关键因素，推理同时对吞吐和时延具有较高要求。**投机解码（Speculative Decoding）**作为一种提升大模型推理性能的有效技术，近年来受到广泛关注。投机解码通过引入**辅助模型**（通常为小型语言模型）或结构来预测主模型的输出，从而减少主模型的计算负担，提高推理速度。自SD提出以来，相关研究不断涌现，探索更高效的投机解码方法及其在实际系统中的应用。


## 研究课题
从事大模型推理性能优化投机解码方向上的研究，后期可能会根据实际调研和项目需求进行调整。

**大模型推理性能优化-投机解码**：
Transformer架构的大模型推理的解码（decode phase）每次根据注意力窗口内的所有上文信息，生成一个新token。这个自回归的过程受到访存带宽的限制（**memory-bound，访存密集型**）。**投机解码（Speculative Decoding）**通过引入辅助的**草稿模型**（通常为小型语言模型，**draft model**）或结构来预测主模型的输出的连续若干个token，**目标模型（target model）**只需对草稿模型预测的token进行验证，从而在前向计算中并行生成多个token，减少目标模型的计算负担，提高推理速度。

投机解码的研究领域主要集中在**token预测**的优化上：研究如何设计高效的draft model或预测模块，包括模型结构、参数量、训练方法、预测token数量等，以在保证预测准确率的同时最大化推理速度提升。

在本课题中，我们主要关注从两方面优化投机解码的性能：

## 方向1. 
负责人：王智彬，张中辉（待定）

**通过缓存之前预测的有效信息来提升draft model的预测能力**：通过设计高效的缓存机制，存储和利用之前预测的（包括没有被命中）token及其上下文信息，提升draft model在连续token预测中的准确率和效率。可以优化的方向包括但不限于： a. **缓存更有效的信息**，以提升预测效率； b. **优化检索策略**，如向量检索，文本匹配等的效率。

## 方向2. 
负责人：刘复良

**Draft model结构设计**：优化预测模块结构设计。

你也可以探索其他相关领域，只要总的方向是从投机解码的角度上优化大模型推理性能。

## 推荐论文和项目

- **SpecForge**：SpecForge is an ecosystem project developed by the SGLang team. It is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference. Learn more: https://docs.sglang.ai/SpecForge/。

- **Accelerating Large Language Model Decoding with Speculative Sampling (2023)**  

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** （SD方向的著名论文，与上述方向2相关）

- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** （还包括EAGLE-2, 3系列，EAGLE-3是当前业界落地的SOTA工作

- **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**

- **Better & Faster Large Language Models via Multi-token Prediction**

- **Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling**  （与上述方向1强相关）

此外，还包括Deepseek等的MTP等优化方案，可以扩展阅读。

