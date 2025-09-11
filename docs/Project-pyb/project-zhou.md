# 大模型训练与推理性能优化

## 项目背景

随着大规模语言模型（LLM）在各类自然语言处理任务中的广泛应用，模型的训练与推理效率成为制约其落地和推广的关键因素。如何对大模型的训练与推理过程进行性能建模，并据此进行系统优化，是当前学术界和工业界关注的热点问题。

## 已有工作

- **Accelerating Model Training on Ascend Chips: An Industrial System for Profiling, Analysis and Optimization (ATC 2025)**
- **Squeezing Operator Performance Potential for the Ascend Architecture (ASPLOS 2025)**

## 研究课题
初步计划从事以下研究，后期可能会根据实际调研和项目需求进行调整。

**大模型推理性能建模与自动调优**：
在大模型推理部署过程中，性能调优面临以下几个方面的挑战：
- 首先，性能优化目标并不统一。具体来说，推理系统需要在延迟和吞吐之间进行权衡，如何根据不同的业务需求，合理定义并动态调整优化目标，缺乏通用方案。
- 其次，性能影响因素复杂且高度耦合。推理效率受到多种因素的共同作用，包括负载特征（输入/输出长度分布，请求数量等）、模型与量化（模型架构，参数量，量化精度）、系统与硬件配置（加速器数量与类型，内存大小，带宽）、软件与算法策略（并行策略、批处理调度、KV缓存管理、算子融合、是否分离部署）等，缺乏全面系统的分析以确定性能瓶颈。
- 最后，当前优化方法低效且成本高昂。目前业界普遍依赖专家经验进行手动试错式调优，但存在根本缺陷。实际部署中，影响因素的参数组合呈指数级增长，每一种组合的性能差异巨大。同时每轮参数验证都需要消耗数小时甚至更长的计算。因此该方案难以适应频繁变化的业务需求和系统环境。

总的来说，现有的手动调优难以满足高效、可扩展的大模型推理部署需求。因此亟需一种适应多场景需求、自动识别关键性能瓶颈、智能搜索最优配置组合的推理性能自动寻优方案。

## 推荐论文

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

- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (2024)**

- **MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism (2025)**  

- **LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference (2024)**

- **AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs (2025)**
