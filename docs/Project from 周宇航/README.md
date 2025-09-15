# 大模型训练与推理性能优化

## 项目背景

随着大规模语言模型（LLM）在各类自然语言处理任务中的广泛应用，模型的训练与推理效率成为制约其落地和推广的关键因素。如何对大模型的训练与推理过程进行性能建模，并据此进行系统优化，是当前学术界和工业界关注的热点问题。

## 已有工作

- **Accelerating Model Training on Ascend Chips: An Industrial System for Profiling, Analysis and Optimization (ATC 2025)**
- **Squeezing Operator Performance Potential for the Ascend Architecture (ASPLOS 2025)**

## 研究课题
初步计划从事以下研究，后期可能会根据实际调研和项目需求进行调整。

**大模型推理性能建模与自动调优**：
在大模型推理部署过程中，性能调优面临以下主要挑战：

- **优化目标多样且动态**：推理系统需在延迟与吞吐之间权衡，不同业务场景下的优化目标各异，缺乏通用且可动态调整的优化方案。
- **影响因素复杂且高度耦合**：推理性能受多方面影响，包括：
    - 负载特征（如输入/输出长度分布、请求数量等）
    - 模型与量化（模型结构、参数规模、量化精度等）
    - 系统与硬件配置（加速器类型与数量、内存、带宽等）
    - 软件与算法策略（并行、批处理、KV缓存、算子融合、分离部署等）
    这些因素相互作用，难以系统性分析和定位性能瓶颈。
- **现有调优方法低效且成本高**：业界多依赖专家经验手动调优，参数组合空间巨大，验证周期长，难以适应业务和系统的动态变化。

因此，亟需一种能够自动识别性能瓶颈、智能搜索最优配置、适应多场景需求的大模型推理性能自动调优方案。

## 推荐书籍/博客
- 《机器学习系统：设计和实现》
- [MLSystem 入坑指南 —— 摸黑干活](https://fazzie-key.cool/2023/02/21/MLsys/)
- [2025入坑ML sys 求意见? - 小松的回答 - 知乎](https://www.zhihu.com/question/7717321708/answer/1904210395952033872)
- [2024年MLSys推荐关注列表 - 系统篇 - JerryYin777的文章 - 知乎](https://zhuanlan.zhihu.com/p/13621083399)

## 推荐视频资源（科普向）
- [Transformer架构](https://space.bilibili.com/517221395/lists/2306922?type=season)
- [分布式并行](https://space.bilibili.com/517221395/lists/2646919?type=season)
- [集合通信](https://space.bilibili.com/517221395/lists/3130927?type=season)
- [GPU原理](https://space.bilibili.com/517221395/lists/1282451?type=season)
- [GPU详解](https://space.bilibili.com/517221395/lists/1388713?type=season)

## 推荐论文

### 综述
- **Taming the Titans: A Survey of Efficient LLM Inference Serving (2025)**

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
