# 课程介绍：基于大型语言模型的现实世界系统中的安全（项目式）

> 本课程围绕 **LLM 在真实系统中的安全风险与工程化防护** 展开。

## 个人简介

- **姓名:** 徐闽泽
- **联系方式:** 745231847 AT qq.com (QQ 号也可)

## 主题简述（为何需要“LLM 系统安全”）
- **幻觉（Hallucination）**：模型在缺乏证据时生成看似可信但错误的内容，若直接用于自动化决策或工具链将带来实际风险。  
- **指令/数据边界模糊 → 指令注入（Prompt Injection）**：在 RAG、浏览器/邮箱读取、插件/工具链调用等场景，**外部数据中的隐藏指令**可能被模型误当作“要执行的命令”，引发直接或**间接指令注入**、**提示泄露（Prompt Leaking）**等。  
- **调用过程中的隐私暴露**：云端推理与 RAG 检索会产生**查询隐私**、**向量嵌入反演**、**知识库成员推断**等问题，导致企业/个人敏感数据泄露。  
- **其他相关问题**：模型/Agent 越权执行、工具授权与能力委托、供应链与插件生态风险、数据/知识库投毒、可观测与取证缺失、评测与复现实验基准不足等。

---

## 主要研究方向与课题介绍

### MCP（Model Context Protocol）中的安全性问题
**背景与问题**  
MCP 将模型与外部工具/数据源解耦并标准化连接，但也把**外部内容引入模型上下文与能力边界**：  
- 间接指令注入（来自被检索网页/文档/工具响应）、提示泄露与机密外传、能力滥用（高危工具串联）、跨会话上下文污染、认证与最小权限配置不当等。  

**推荐阅读（主流期刊/会议）**  
- Liu et al., **USENIX Security 2024**. *Formalizing and Benchmarking Prompt Injection Attacks and Defenses*. [论文链接](https://www.usenix.org/system/files/usenixsecurity24-liu-yupei.pdf)  
- Chen et al., **USENIX Security 2025**. *StruQ: Defending Against Prompt Injection with Structured Queries*. [论文链接](https://www.usenix.org/system/files/usenixsecurity25-chen-sizhe.pdf)  
- Hui et al., **ACM CCS 2024**. *PLeak: Prompt Leaking Attacks against Large Language Model Applications*. [论文链接](https://dl.acm.org/doi/10.1145/3658644.3670370)

---

### RAG（Retrieval-Augmented Generation）中的数据隐私问题
**背景与问题**  
RAG 将外部知识接入生成流程，带来以下隐私挑战：  
- **知识库泄露**：对私有库的**内容抽取/成员推断**（e.g., “某条数据是否在库中？”）；  
- **嵌入隐私**：向量表征可能被**嵌入反演**恢复原文；  
- **查询隐私**：云端 RAG 可能泄露用户意图与相关文档分布。  

**推荐阅读（主流期刊/会议）**  
- Zeng et al., **ACL Findings 2024**. *The Good and the Bad: Exploring Privacy Issues in RAG*. [论文链接](https://aclanthology.org/2024.findings-acl.267/)  
- Cheng et al., **ACL Findings 2025**. *RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service*。[论文链接](https://aclanthology.org/2025.findings-acl.197/)  
- Wang et al., **Science China Information Sciences 2025**. *RAG-leaks: Difficulty-Calibrated Membership Inference Attacks on RAG*. [论文链接](https://link.springer.com/article/10.1007/s11432-024-4441-4)  

---

### LLM 驱动的 AI Agents 中的其他安全性问题
**背景与问题**  
LLM Agents 将“感知—规划—工具执行—反馈—记忆”闭环化，并使用浏览器、文件系统、代码解释器、企业 API 等外部能力完成开放环境任务；因此暴露出更复杂的**系统化攻击面**：  
- **工具滥用与越权执行**：高危工具（执行代码/脚本、shell、财务/法务系统 API）被诱导调用会产生真实世界副作用（数据改写、越权访问、资金转移等）。  
- **（间接）指令注入与提示泄露**：来自网页/文档/第三方服务响应中的恶意指令可劫持代理目标或窃取系统提示/密钥。  
- **记忆与知识库投毒**：被污染内容写入长期记忆后，在后续决策与工具调用时被持续放大。  
- **后门与策略植入**：在多步任务或多代理协作中，隐蔽触发条件可改变代理策略与目标（goal hijacking）。  
- **供应链风险**：页面脚本、下载文件、第三方扩展/插件与远端 API 引入新的入口，放大凭据泄露、会话劫持与数据外传风险。  
- **评测与取证困难**：长时序、多回合、多工具链路导致复现、问责与合规审计难度高。

**推荐阅读（主流期刊/会议）**  
- Yang et al., **NeurIPS 2024**. *Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents*. [论文链接](https://neurips.cc/virtual/2024/poster/95425)  
- Deng et al., **ACM Computing Surveys 2025**. *AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways*. [论文链接](https://dl.acm.org/doi/10.1145/3716628)  
- Huang et al., **ACL 2025**. *Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization*. [论文链接](https://aclanthology.org/2025.acl-long.290.pdf)

---

### 任何其他你感兴趣且相关的问题！！！
