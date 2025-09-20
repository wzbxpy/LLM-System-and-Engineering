# Accelerating Collective Communication with Better Congestion Control from 崔正奇

## 联系方式

- **负责人**：[崔正奇]()
- **邮箱**: 522024330010@smail.nju.edu.cn

---

## 项目背景

随着大规模分布式训练（例如 GPT、LLM 等模型）的普及，**集合通信 (Collective Communication)** 成为了性能瓶颈之一。常见操作如 `AllReduce` 、`AlltoAll`和 `Broadcast` 占据了训练中大部分的网络开销。  
近年来，研究者尝试通过 **在网计算 (In-Network Computing, INC)** 来减少通信开销，加速训练过程。

- **ATP (NSDI’21)**：提出利用交换机进行梯度聚合，减少带宽消耗。
- **A2TP (EuroSys’23)**：在 ATP 基础上，设计了感知聚合器 (Aggregator-aware) 的拥塞管理方案。
- **In-Network Aggregation with Transport Transparency (ASPLOS’23)**：提出透明化的在网聚合机制，使得应用几乎无需修改即可受益。

然而，现有方法在 **拥塞管理** 上仍然不足，尤其是当交换机寄存器不足时，会导致热点端口或链路拥塞，直接影响训练速度。

---

## 已有工作

### ATP: In-network Aggregation for Multi-tenant Learning (NSDI 2021)
- 首次提出在交换机上执行梯度聚合，减少通信流量。
- 缺陷：缺乏高性能拥塞管理方案和动态调度的支持。

### A2TP: Aggregator-aware In-network Aggregation (EuroSys 2023)
- 考虑多租户场景，引入聚合器感知的调度。
- 问题：依赖交换机寄存器存储状态，当寄存器不足时，端侧激进地减速，导致性能下降明显。

### In-Network Aggregation with Transport Transparency (ASPLOS 2023)
- 将在网聚合抽象为对应用透明的传输服务。
- 优点：无需修改上层框架即可受益。
- 缺陷：仍未解决底层的拥塞控制挑战。

---

## 项目内容

本项目旨在突破 **A2TP 的局限性**，在寄存器不足的场景下，通过**改进拥塞管理**来提升集合通信效率。具体包括：

- **问题动机**  
  当交换机寄存器不足时，无法同时存储所有流的聚合状态，导致部分流退化为常规传输，造成严重拥塞。
  
- **研究方向**  
  - 基于网络信号（如RTT、ECN）设计新的 **精细化拥塞控制算法**  
  - 动态调度不同租户/任务之间的流量，提升带宽利用率  
  - 结合 INC 与传统拥塞控制（如 DCQCN, TIMELY），在有限硬件资源下实现更高效的集合通信  

---

## 研究课题

 **新型拥塞控制机制**  
   - 结合 ECN 与 RTT 信号，避免单纯依赖寄存器，避免寄存器成为通信瓶颈  
   - 针对新型通信协议与架构（UEC, Ultra Enthernet Consortium），探索并设计通用的拥塞控制机制

---
## 推荐博客
- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇）](https://zhuanlan.zhihu.com/p/681154742)
  - 这篇博客通过图解阐述模型并行、数据并行、专家并行的通信过程，非常适合新手理解分布式训练的基本过程
- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（源码解读篇）](https://zhuanlan.zhihu.com/p/681692152)

---
## 推荐阅读/参考文章

### 在网计算相关
- [ATP: In-network Aggregation for Multi-tenant Learning (NSDI 2021)](https://dl.acm.org/doi/10.1145/3477132.3483560)
- [A2TP: Aggregator-aware In-network Aggregation (EuroSys 2023)](https://dl.acm.org/doi/10.1145/3552326.3567477)
- [In-Network Aggregation with Transport Transparency (ASPLOS 2023)](https://dl.acm.org/doi/10.1145/3575693.3578821)

### 分布式训练
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019)](https://arxiv.org/pdf/1909.08053)
- [Pipeline Parallelism Overview](https://arxiv.org/pdf/2104.04473)

### 拥塞控制
- DCQCN: Data Center Quantized Congestion Notification
- [TIMELY: RTT-based Congestion Control for Datacenter Networks](https://dl.acm.org/doi/10.1145/2815675.2815680)
  - 两篇文章是数据中心网络的经典文章，DCQCN使用ECN作为拥塞信号，TIMELY使用RTT作为拥塞信号，是理解两种信号及应用方式的必读之作

