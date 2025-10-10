# 论文列表

## 模型架构（李世鹏）

1. **Attention Is All You Need** (2017)
    - **作者**: Vaswani et al. (Google)
    - **核心贡献**: 提出了**Transformer**架构，完全基于自注意力（Self-Attention）机制，摒弃了传统的循环（RNN）和卷积（CNN）结构。这是迄今为止大模型领域最基础、最核心的论文，为后续所有LLM奠定了架构基础，是无可争议的奠基之作，引用量超过10万次。

2. **LLaMA: Open and Efficient Foundation Language Models** (2023)
    - **作者**: Hugo Touvron et al. (Meta AI)
    - **核心贡献**: 对Transformer进行了一系列**精妙的优化设计**。包括使用**RMSNorm**、**SwiGLU**激活函数、**RoPE**位置编码等。证明了通过更好的数据、干净的架构和高效的训练，参数量更小的模型可以超越更大规模的模型，引发了"小而精"模型和开源LLM的浪潮，是许多后续模型的基石。

3. **RetNet: Retentive Network: A Successor to Transformer for Large Language Models** (2023)
    - **作者**: Microsoft Research
    - **核心贡献**: 提出了一种名为**Retention（保留）**机制的新架构，旨在同时实现**训练并行化、低成本推理和良好的性能**，解决Transformer在推理时计算成本高的问题。

4. **Mixture of Experts (MoE)**
    - **代表论文**: **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** (2020) & **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (2021)
    - **作者**: Google Research
    - **核心贡献**: 将**混合专家（MoE）**系统成功引入到Transformer架构中。通过让每个输入只激活一部分网络（"专家"），实现了模型参数量巨幅增长（如万亿参数）而计算成本基本不变，解决了纯稠密模型scaling的瓶颈，是目前超大规模模型（如DeepSeek-R1）的核心技术，是scaling law下的关键架构创新。

5. **RWKV: Reinventing RNNs for the Transformer Era** (2022-ongoing)
    - **作者**: Bo Peng et al.
    - **核心贡献**: 提出了一种新颖的架构，**将Transformer的高效并行训练与RNN的低成本线性推理相结合**。它不再是传统的Transformer，而是一种"RNN Transformer"，推理时不需要K-V Cache，内存占用极低，且支持无限上下文；在开源社区非常热门，被认为是挑战Transformer统治地位的有力候选之一，特别适合资源受限的场景。

6. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2023)
    - **作者**: Albert Gu & Tri Dao (CMU, Stanford)
    - **核心贡献**: 基于**结构化状态空间模型（SSM）**，提出了**Mamba**架构。它引入了"选择性"机制，使模型能够根据输入内容动态地选择性地传递或遗忘信息，在长序列建模上性能显著超越Transformer，且具有线性时间的推理效率，引发了SSM相关研究的热潮。

7. **The Gemini 1.5 Technical Report** (2024)
    - **作者**: Google DeepMind
    - **核心贡献**: 其架构核心是**MoE Transformer**，但最大的热点在于其引入了**全新的高效注意力机制**（可能是MLA，Multi-Head Latent Attention），从而实现了**百万级别的超长上下文（Context Window）**支持，并且性能衰减极小；重新定义了上下文长度的可能性，其背后的高效架构设计是当前的研究前沿和热点。

8. **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models** (2024)
    - **作者**: Google DeepMind
    - **核心贡献**: 提出了一种**将线性递归（RNN）与局部注意力（Local Attention）相结合**的新架构。它在保持高效并行训练的同时，实现了媲美Transformer的性能，并且推理速度更快、内存效率更高，展示了混合架构的强大潜力。

9. **Vision Transformer (ViT)** (2020)
    - **作者**: Dosovitskiy et al. (Google)
    - **核心贡献**: 证明了**纯Transformer架构在计算机视觉任务上同样可以取得state-of-the-art的性能**。它将图像切分为Patch序列，然后直接作为输入送入标准Transformer Encoder。这项工作打破了CV和NLP的架构壁垒，开启了多模态模型的基础；开创了视觉任务的新范式，是ViT、DeiT、Swin Transformer等无数工作的起点。

10. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)
     - **作者**: Devlin et al. (Google)
     - **核心贡献**: 虽然基于Transformer Encoder，但其架构设计上的热点在于**掩码语言模型（MLM）预训练目标**和**双向上下文编码**。这种预训练架构设计使得模型能生成深度的上下文词表征，在NLP任务上取得了颠覆性的效果；与GPT（Decoder-only）并驾齐驱，开创了NLP的预训练时代，其架构思想影响深远。

## 调度
**todo by lsp**

## 分布式训练、推理（周宇航）
### 分布式训练方向

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019)**  

- **Megatron-LM: Reducing Activation Recomputation in Large Transformer Models (2021)**

- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020)**  
   
- **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning (2022)**  

- **FSDP: Fully Sharded Data Parallel (2023)**  

- **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs (2024)**

- **Minder: Faulty Machine Detection for Large-scale Distributed Model Training (2025)**

- **Towards LLM-Based Failure Localization in Production-Scale Networks (2025)**

- **Recycle: Resilient Training of Large DNNs using Pipeline Adaptation (2024)**

---

### 分布式推理方向
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)**  

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)**  

- **Orca: A Distributed Serving System for Transformer-Based Generative Models (2023)**  

- **Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)**  

- **Accelerating Large Language Model Decoding with Speculative Sampling (2023)**
  
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (2024)**  

- **DistServe: Disaggregating Prefill and Decoding for LLM Serving (2024)**  

- **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (2025)**

- **MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism (2025)**  

- **LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference (2024)**

- **AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs (2025)**

---

## 向量数据库方向（陈力峥，赵可泰）

- **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (2018)**  
  
- **DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node (2019)**  
  
- **A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search (2021)**  
  
- **AnalyticDB-V: A Hybrid Analytical Engine Towards Query Fusion for Structured and Unstructured Data (2020)**  
  
- **Navigable Proximity Graph-Driven Native Hybrid Queries with Structured and Unstructured Constraints (2022)**  
  
- **Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters (2023)**  
  
- **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data (2024)**  
  
- **Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search (2025)**  
  
- **FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search (2021)**  
  
- **In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search (2025)**  


---

## 大模型内存管理（王梓博）

- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**

- **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**

- **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**

---


## 卸载推理系统

- **FlexGen: high-throughput generative inference of large language models with a single GPU (2023)**

- **HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices (2024)**

- **NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference (2024)**
  
- **SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices (2024)**

- **MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs (2024)**

- **Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline (2025)**
  
- **FlexInfer: Flexible LLM Inference with CPU Computations (2025)**

---

## 强化学习

- **ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation (2025)**
  
- **HybridFlow: A Flexible and Efficient RLHF Framework (2025)**
  
- **Optimizing RLHF Training for Large Language Models with Stage Fusion (2025)**

---

## 大模型 for 通信（彭于波）
### 语义通信方向

- **Large AI model empowered multimodal semantic communications (2024)**  

- **Large language model enhanced multi-agent systems for 6G communications (2024)**

- **Large AI model-based semantic communications (2024)**  
   
- **Large generative model assisted 3D semantic communication (2024)**  

---

### 多模态通感一体化方向
- **Simac: A semantic-driven integrated multimodal sensing and communication framework (2025)**
---
