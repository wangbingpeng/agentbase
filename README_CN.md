# AgentBase — AI 代理的上下文数据库

[English](README.md)

> **Context DB = Memory + Resource + Skill + Temporal + Session + Observability**

AgentBase 是一个为 AI 代理设计的开源上下文数据库。它提供统一的、可查询的存储，涵盖记忆、资源、技能、时序知识、会话历史和可观测性——全部基于 SQLite 构建，零外部基础设施依赖。

**LongMemEval Overall: 73.2% — 完全开源方案中排名第一**

## 基准测试 — LongMemEval 对比

LongMemEval 是长期记忆系统的事实标准基准，在多轮对话场景下评估 5 种题型的检索准确率（约 500 道题）。

### 分题型准确率

| 题型 | AgentBase | Mem0 OSS | Mem0 Pro | OMEGA | Zep/Graphiti |
|---|---|---|---|---|---|
| 单会话 (用户) | 90.0% | 94.3% | 97.1% | 99.2% | — |
| 单会话 (助手) | **92.9%** | 46.4% | 100.0% | 99.2% | — |
| 单会话 (偏好) | 66.7% | 76.7% | 96.7% | 100.0% | — |
| 知识更新 | **91.0%** | 79.5% | 96.2% | 96.2% | — |
| 时序推理 | 62.4% | 51.1% | 93.2% | 94.0% | 63.8% |
| 多会话 | 57.9% | 70.7% | 86.5% | 83.5% | — |
| **总计** | **73.2%** | 67.8% | 93.4% | 95.4% | 63.8% |

> 数据来源：Mem0 (docs.mem0.ai, 2026.3)，OMEGA (omegamax.co, 2026.2)，Zep (vectorize.io, 2026)，AgentBase 本地评测。

### AgentBase vs Mem0 OSS — 公平的开源对比

Mem0 Pro (93.4%) 是**付费托管平台**，包含专有优化。Mem0 官方文档声明：*"分数反映了 Mem0 的托管平台，其中包含开源 SDK 中不提供的专有优化。"* 对于开源用户，公平的对比对象是 Mem0 OSS：

| 维度 | AgentBase | Mem0 OSS | 差距 |
|---|---|---|---|
| 整体准确率 | **73.2%** | 67.8% | **+5.4pp** |
| 知识更新 | **91.0%** | 79.5% | **+11.5pp** |
| 时序推理 | **62.4%** | 51.1% | **+11.3pp** |
| 单会话 (助手) | **92.9%** | 46.4% | **+46.5pp** |
| 外部依赖 | 无 (SQLite) | 需 Qdrant | 更轻量 |
| Ingest 阶段 LLM 调用 | **0** | 每条消息 | 零 ingest 成本 |

Mem0 OSS 在单会话（助手）上仅 46.4%——这是对助手角色信息检索的严重盲区。AgentBase 通过会话摘要提取 + 全轮次存储覆盖了此场景。

### AgentBase vs Zep/Graphiti — 轻量 vs 重图

| 维度 | AgentBase | Zep/Graphiti |
|---|---|---|
| 整体准确率 | **73.2%** | 63.8% | **+9.4pp** |
| 时序推理 | 62.4% | 63.8% | ~持平 |
| 外部数据库 | 仅 SQLite | Neo4j / 图数据库 |
| 部署方式 | `pip install` | Docker + 图数据库 |

### 成本效率 — 独特的零 LLM Ingest 优势

| 指标 | AgentBase | Mem0 Pro | OMEGA |
|---|---|---|---|
| 平均 Token / 查询 | ~3,500 | ~6,787 | ~7,000+ |
| 外部依赖 | 无 | Qdrant | SQLite |
| Ingest 阶段 LLM 调用 | **0** | 每条消息 | 每条消息 |
| GPU 需求 | 无 | 无 | ONNX (CPU) |

AgentBase 的**零 LLM ingest** 是独特优势：500 题的整个 ingest 阶段不消耗任何 LLM token，而 Mem0 和 OMEGA 每条消息都需要 LLM 调用。

### 诚实说明

- Mem0 Pro (93.4%) 和 OMEGA (95.4%) 在绝对准确率上领先——两者都是付费/托管平台，使用专有优化和更强的回答模型。
- OMEGA 和 Mastra 使用 GPT-4.1 / GPT-4o 作为回答模型；AgentBase 使用 qwen-plus（约 GPT-4o-mini 水平）。模型能力差异约 5-10pp。
- AgentBase 的多会话 (57.9%) 和时序推理 (62.4%) 仍有较大提升空间，主要受限于 IDH（智能去重与硬化）过度拒绝。
- **在完全开源、自托管、零依赖的方案中，AgentBase 取得了最高的 LongMemEval 分数。**

## 创新性

AgentBase 引入多项业界首创技术，使其在 Mem0、Zep/Graphiti、LangChain Memory 等开源记忆系统中脱颖而出。

### 1. 零 LLM Ingest — 唯一的零成本 Ingest 管线

其他所有记忆框架在 ingest 阶段都需要 LLM 调用：Mem0 对每条消息调用 LLM 提取事实；Zep 用 LLM 做情节摘要；OMEGA 每条消息都运行 LLM。AgentBase 通过**完全本地规则化管线**实现结构化记忆提取：

| Ingest 步骤 | AgentBase | Mem0 OSS | Zep/Graphiti |
|---|---|---|---|
| 会话摘要 | 规则（LLM 可选） | 每会话 LLM | 每会话 LLM |
| 实体提取 | spaCy + 正则（本地） | 每条消息 LLM | 每条消息 LLM |
| 事实提取 | 模式匹配（本地） | 每条消息 LLM | 每条消息 LLM |
| 层级生成 | 截断降级（LLM 可选） | 无 | 无 |
| 去重与硬化 | IDH 规则（本地） | 无 | 无 |

**影响**：500 题 × 0 次 LLM 调用 = **ingest 阶段消耗零 token**，而 Mem0 消耗约 ~6,787 token/查询。这消除了 ingest 延迟、网络依赖、速率限制和成本——支持边缘部署、离线使用和 CI 环境。

### 2. 三路召回 (FTS + 向量 + NER) — 业界首创

现有框架使用单路（Mem0：仅通过 Qdrant 向量检索）或双路（Zep：图 + 向量）检索。AgentBase 添加了**第三个信号——NER 实体提升**——作为 FTS+向量结果上的精度放大器：

```
  ┌─────────┐   ┌─────────┐   ┌─────────────┐
  │  FTS5   │   │  向量   │   │  NER 实体   │
  │ (BM25)  │   │(cosine) │   │   提升      │
  └────┬────┘   └────┬────┘   └──────┬──────┘
       │   RRF 融合    │              │
       └────────┬──────┘              │
                ▼    ◄── 分数提升 ──┘
          融合结果（NER 永远不添加新结果）
```

**为什么重要**：Mem0 OSS 的单会话（助手）得分仅 46.4%——这是对助手角色信息的严重盲区。纯向量搜索会遗漏精确实体匹配（"Hawaii" 在语义上不接近 "trip"）。AgentBase 的 FTS5 捕获精确关键词 + NER 标签 (`ner_Hawaii`) 提升相关结果——这是 AgentBase 在同一指标上达到 92.9% 的技术根因。

**三级 NER 匹配**（防止稀释）：
- 强匹配（标签精确匹配）：`score × 1.3` — `matched_by = "ner+hybrid"`
- 中匹配（NER 标签 + 内容匹配）：`score × 1.24`
- 弱匹配（仅内容匹配）：`score × 1.15`

### 3. 查询类型感知检索 — 5 种意图策略自动检测

其他框架对所有查询一视同仁。AgentBase 的 `IntentAnalyzer` 检测 5 种查询类型并应用专门的后处理：

| 查询类型 | 创新点 | 行业通行做法 |
|---|---|---|
| **时序推理** | 自动解析日期表达式 → 填充 `date_from`/`date_to` 过滤器 (D5) | 无时间感知 |
| **知识更新** | 双半衰期：通用 7 天 / 知识更新 30 天（强时效偏好） | 无时效区分 |
| **多会话** | 从 DB 跨会话补全 + 按会话去重 | 仅单会话 |
| **偏好** | 多级信号：用户角色 (1.8×) + 偏好分类 (2.0×) + 隐含指标 (1.3×) | 无偏好感知 |
| **聚合** | D2：自动提升 `top_k` 20→120 实现穷尽召回 + 枚举提示后缀 | 固定 top_k |

**聚合检测** (D2) 尤其具有前瞻性："多少"类问题需要穷尽召回而非 top-k 截断。没有这个机制，聚合查询会系统性低计数。

### 4. 会话协同检索 (D1) — 解决“上下文丢失”问题

当检索命中对话中的某条消息时，周围上下文至关重要。AgentBase 通过 `session_memory_links` 自动扩展结果：

- **基于链接的扩展**：跟随会话提交时写入的 `session_memory_links`（FK 完整性保证）查找协同相关条目
- **轮次感知门控**：仅对 ≥2 轮会话激活——防止单轮噪声
- **预算感知**：协同检索的条目不会超过调用方的 `token_budget`

Mem0 将每条消息提取为孤立事实，丢失对话上下文。LangChain 保留最近 k 轮，无语义关联。AgentBase 是唯一在数据库层面维护**会话与提取记忆双向链接**的框架。

### 5. 三层渐进内容 (L0/L1/L2) 与确定性加载

| 层级 | 内容 | 长度限制 | 用途 |
|---|---|---|---|
| L0 | 摘要/标题 | ≤50 字符 | 粗筛、列表展示 |
| L1 | 概述 | ≤300 字符 | 精排、预览 |
| L2 | 全文 | 无限制 | 最终回答 |

**超越简单摘要的创新**：
- **确定性自动选择**：6 条确定性规则（top_k>20→L0, budget<1000→L0, hierarchical→L1 等）——不含随机性
- **渐进搜索**：`strategy=hierarchical` 搜索 L0 (3× top_k) → L1 (2× top_k) → L2（仅最终结果），I/O 减少约 60%
- **截断降级**：`fallback_to_truncation=true` 在无 LLM 时通过截断生成 L0/L1——保证功能完整

### 6. 纯 SQLite 零依赖架构

| 框架 | 外部依赖 | 部署方式 |
|---|---|---|
| **AgentBase** | **无** (SQLite + FTS5 内建) | `pip install` |
| Mem0 OSS | Qdrant (向量数据库) | Docker / 云服务 |
| Zep/Graphiti | Neo4j (图数据库) | Docker + 图数据库 |
| LangChain | 无（但无检索能力） | pip install（功能缺失） |

FTS5 是 SQLite 内建扩展；`sqlite-vec` 是可选增强。这意味着 AgentBase 可运行在嵌入式设备、CI 环境甚至浏览器 (Pyodide) 中。

### 7. 双语 NER 降级链

```
spaCy (英文) → spaCy (中文) → 正则回退（中文地名/机构 + 英文大写名称/数量） → 跳过
```

没有其他开源记忆框架具备本地 NER 能力。Mem0 和 Zep 将所有实体识别委托给 LLM，使 NER 在无 LLM 时不可用。

### 8. 多框架统一记忆后端

全部 5 个适配器共享同一个 SQLite 数据库——这是其他框架不具备的“统一记忆后端”能力：

- `Mem0Adapter` — 替换 `Memory()` 一行搞定
- `LangChainMemoryAdapter` — 鸭子类型兼容 `BaseChatMemory`
- `AgentBaseChatStore` — 实现 LlamaIndex `BaseChatStore`
- `OpenAIAssistantAdapter` — 映射 Thread→Session
- `MinimalAdapter` — 三方法 API (remember/recall/forget)

### 9. 全栈可观测性

| 层级 | 能力 | 细节 |
|---|---|---|
| **追踪持久化** | `retrieval_traces` + `trace_steps` 表 | 每步延迟、候选数、模型名、缓存命中 |
| **Web 仪表盘** | 6 项可视化 | 时间线、热力图、分类旭日图、新鲜度分布、标签云、活动流 |
| **调试 API** | `trace_session()`、`entity_graph()`、`diff_entries()` | 编程式内省 |

Mem0 OSS 无可观测性界面；Zep 有基础仪表盘但无检索追踪；LangChain 无。

---

**创新矩阵总结**：

| 创新 | 业界首创 | 核心价值 |
|---|---|---|
| 零 LLM Ingest | ✅ | 零成本、零延迟、零依赖的 ingest |
| 三路召回 (FTS+Vec+NER) | ✅ | 关键词 + 语义 + 实体三维覆盖 |
| 查询类型感知策略 | ✅ | 自动检测 5 种意图的专门处理 |
| 会话协同检索 (D1) | ✅ | 解决命中但丢上下文问题 |
| 三层渐进搜索 | ✅ | 60% I/O 削减，确定性层级选择 |
| 纯 SQLite 零依赖 | ✅ | pip install 即用，无需 Docker/云 |
| 双语 NER 降级链 | ✅ | 无需 LLM 的本地实体识别 |
| 多框架统一后端 | ✅ | 5 个适配器共享同一数据库 |
| 全栈可观测性 | ✅ | 追踪持久化 + Web 仪表盘 + 调试 API |

## 功能特性

- **记忆 (Memory)** — 存储和检索代理记忆（偏好、事实、流程）
- **资源 (Resource)** — 管理外部资源（URL、文档、API）
- **技能 (Skill)** — 注册和发现工具能力
- **时序知识图谱** — 实体-关系图，支持时间感知的事实追踪
- **会话管理** — 多轮对话跟踪与记忆提取
- **Web 仪表盘** — 可视化可观测性：时间线、热力图、检索追踪、分类旭日图、新鲜度分布、标签云
- **三路混合检索** — FTS5 全文 + sqlite-vec 向量 + NER 实体提升，RRF 融合
- **三层内容** (L0/L1/L2) — 渐进式细节层级，高效检索
- **多代理作用域** — 全局、代理、项目、会话级隔离
- **框架适配器** — 兼容 Mem0 / LangChain / LlamaIndex / OpenAI Assistants，即插即用

## 快速开始

### 安装

```bash
# 使用 uv（推荐）
uv sync --all-packages --all-extras --dev

# 或使用 pip
pip install -e packages/agentbase-core[all]
pip install -e packages/agentbase-sdk
pip install -e packages/agentbase-cli
pip install -e packages/agentbase-mcp
pip install -e packages/agentbase-web
```

### Python SDK

```python
import asyncio
from agentbase import AgentBase

async def main():
    # 初始化
    db = AgentBase(path="./my_agent.db")
    await db.initialize()

    # 添加记忆
    await db.add_memory("用户偏好 Python 3.12", category="preference", tags=["python"])

    # 搜索
    results = await db.find("Python 偏好", top_k=5)
    for r in results:
        print(f"[{r.entry.context_type.value}] {r.entry.l2_full}")

    await db.close()

asyncio.run(main())
```

### 命令行

```bash
# 初始化数据库
agentbase init --data-dir ./data

# 添加条目
agentbase add "用户偏好暗色模式" --type memory --tags "preference,dark-mode"

# 搜索
agentbase find "用户偏好" --top-k 5

# 查看条目
agentbase get <entry-id>

# 会话管理
agentbase session create --agent-id my-agent
agentbase session add-message <session-id> --role user --content "你好"
agentbase session commit <session-id>
```

### MCP 服务器

```bash
# 启动 MCP 服务器（stdio 传输）
agentbase-mcp
```

### Web 仪表盘

```bash
# 启动 Web 仪表盘
agentbase-web ./my_agent.db 8080
```

## 框架适配器

AgentBase 为主流 AI 框架提供了即插即用的适配器，只需一行代码即可替换记忆后端，无需修改现有代码。

| 适配器 | 框架 | 兼容接口 | 关键方法 |
|---|---|---|---|
| `Mem0Adapter` | [Mem0](https://github.com/mem0ai/mem0) | `mem0.Memory` | `add`、`search`、`get_all`、`update`、`delete` |
| `LangChainMemoryAdapter` | [LangChain](https://github.com/langchain-ai/langchain) | `BaseChatMemory` | `save_context`、`load_memory_variables`、`clear` |
| `AgentBaseChatStore` | [LlamaIndex](https://github.com/run-llama/llama_index) | `BaseChatStore` | `add_message`、`get_messages`、`delete_message` |
| `OpenAIAssistantAdapter` | [OpenAI Assistants](https://platform.openai.com/docs/assistants) | `beta.threads` | `create_thread`、`create_message`、`list_messages` |
| `MinimalAdapter` | — | 三方法 API | `remember`、`recall`、`forget` |

所有适配器共享**同一个 SQLite 数据库**——你可以在单个 `AgentBase` 实例上同时使用多个适配器，互不干扰。

### 从 Mem0 迁移

```python
from agentbase import AgentBase
from agentbase.adapters import Mem0Adapter

db = AgentBase(path="./mem.db")
await db.initialize()

# 替换前：m = Memory()
# 替换后：m = Mem0Adapter(db)
m = Mem0Adapter(db)
m.add("我喜欢披萨", user_id="alice")
results = m.search("食物偏好", user_id="alice")
```

### LangChain 集成

```python
from agentbase import AgentBase
from agentbase.adapters import LangChainMemoryAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

memory = LangChainMemoryAdapter(db, owner_id="alice")
memory.save_context({"input": "你好"}, {"output": "你好呀！"})
history = memory.load_memory_variables({"input": "你好"})
```

### LlamaIndex ChatStore

```python
from agentbase import AgentBase
from agentbase.adapters.llamaindex import AgentBaseChatStore
from llama_index.core.memory import ChatMemoryBuffer

db = AgentBase(path="./mem.db")
await db.initialize()

chat_store = AgentBaseChatStore(db)
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user_alice",
)
```

### OpenAI Assistants API

```python
from agentbase import AgentBase
from agentbase.adapters import OpenAIAssistantAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

oa = OpenAIAssistantAdapter(db)
thread = oa.create_thread(metadata={"agent_id": "my-agent"})
oa.create_message(thread_id=thread["id"], role="user", content="你好")
messages = oa.list_messages(thread_id=thread["id"])
context = oa.retrieve_context(thread_id=thread["id"], query="用户偏好")
```

### Minimal API（最简集成）

```python
from agentbase import AgentBase
from agentbase.adapters import MinimalAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

mem = MinimalAdapter(db)
mem.remember("用户偏好暗色模式", who="alice", tags=["preference"])
results = mem.recall("主题偏好", who="alice")
mem.forget(entry_id="...")

# 也支持异步：
# await mem.aremember(...)
# await mem.arecall(...)
# await mem.aforget(...)
```

## 检索召回架构

AgentBase 使用**三路层级递进检索管线** —— FTS5 + 向量 + NER —— 将关键词精确性、语义理解和实体感知提升相结合。默认情况下，整个 ingest 和检索过程无需调用 LLM。

### 管线概览

```
查询 ──► 规范化 ──► 意图检测 ──► 查询分解
                                   │
              ┌────────────────────┘
              ▼
     ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
     │  FTS5 (BM25) │  │  sqlite-vec  │  │  NER 实体     │
     │   全文搜索   │  │  向量搜索    │  │  提升信号     │
     └──────┬──────┘  └──────┬───────┘  └───────┬───────┘
            │   三路 RRF 融合      │                │
            └──────────┬──────────┘                │
                       ▼    ◄──── NER 分数提升 ──┘
              启发式重排序
            (时效性/置信度/作用域/类型)
                       │
              ┌────────┴────────┐
              ▼                 ▼
        查询类型策略        会话协同检索
        (类型化后处理)     (链接扩展)
              │                 │
              └────────┬────────┘
                       ▼
           加载层级 (L0/L1/L2)
           + Token 预算裁剪
                       │
                       ▼
                    结果
```

### 三路召回：FTS5 + 向量 + NER

AgentBase **并行运行三个独立的召回信号**，融合为统一排名：

| 信号 | 引擎 | 优势 | 默认权重 |
|---|---|---|---|
| **FTS5** | SQLite 内置 BM25 | 精确关键词匹配，零延迟，无需嵌入 | 0.4 |
| **向量** | sqlite-vec (cosine distance) | 语义相似度，处理同义改写和近义词 | 0.6 |
| **NER** | spaCy + 正则双语 NER | 实体感知提升，将查询实体链接到标记条目 | 0.3 |

**NER 如何集成到管线中**（不是独立搜索路径，而是分数提升信号）：

1. **Ingest 阶段** — `NerExtractor` 从每个条目中提取命名实体并标记为 `ner_<实体名>`（如 `ner_Hawaii`、`ner_Python`）。支持双语：spaCy（中/英）+ 正则回退，覆盖中文地名/机构名和英文大写名称/数量。
2. **查询阶段** — 相同的 `NerExtractor` 从查询文本中提取实体。
3. **融合阶段** — 对于每个已有的 FTS+向量结果，如果其 `ner_*` 标签与查询实体匹配：
   - **强匹配**（标签精确匹配）：`score *= (1 + ner_weight)` → `matched_by = "ner+hybrid"`
   - **中匹配**（NER 标记条目 + 内容中包含实体）：`score *= (1 + ner_weight * 0.8)`
   - **弱匹配**（仅内容中包含实体文本）：`score *= (1 + ner_weight * 0.5)`
4. **不稀释** — NER 永远不会在 FTS+向量结果集之外添加新结果，防止无关条目进入管线。

此设计意味着 NER 作为双路召回之上的**精度提升器**，而非可能引入噪声的独立检索路径。

**RRF 融合**（FTS + 向量）：`score(d) = Σ weight_i / (k + rank_i(d))` ——默认 k=60。产生统一排名，同时捕获关键词命中和语义邻居。NER 提升在 RRF 融合**之后**应用，放大实体相关结果。

**优雅降级链**：向量不可用 → 纯 FTS + `degrade_reason` 标记。spaCy 不可用 → 正则 NER 回退。NER 返回空 → 跳过提升，保持 FTS+向量结果不变。

### 零 LLM Ingest 管线

与 Mem0 和 OMEGA 每条消息都需要 LLM 调用不同，AgentBase 通过**本地规则化处理**提取记忆：

1. **会话摘要** — 从对话轮次自动生成会话摘要（LLM 可选，可使用规则）
2. **NER 提取** — 双语 NER：spaCy（中/英）优先 → 正则回退（大写名称、中文地名/机构、数量）。为条目标记 `ner_<实体名>` 用于检索提升。
3. **事实提取** — 基于模式的事实提取，不依赖 LLM
4. **层级生成** — L0 摘要 / L1 中间层 / L2 全文内容（LLM 可选）
5. **智能去重与硬化 (IDH)** — 对相似条目去重并硬化置信度分数

这意味着 500 题的 ingest 阶段消耗 **零 LLM token**，而 Mem0 和 OMEGA 每条查询消耗约 6,000-7,000+ token。

### 查询分解 — 零 LLM

当 LLM 不可用时，`LocalQueryDecomposer` 使用规则策略拆分查询：

1. **模式提取** — 识别 `"多少 X"`、`"哪种 X"`、`"我何时 X"` 并提取关键名词短语
2. **停用词移除** — 双语停用词过滤（中文 + 英文）生成仅关键词子查询
3. **时序 token 增强 (D3)** — 从时序查询中提取日期模式：`"2022年5月"`、`"3周前"`、`"上个月"`、英文日期格式
4. **多子查询搜索** — 每个子查询独立搜索，结果按条目 ID 去重

这以零成本提供了 LLM 意图分解的合理近似。

### 查询类型感知检索

引擎自动检测查询意图并应用专门的后处理策略和调优参数：

| 查询类型 | 检测方式 | 策略 | 关键参数 |
|---|---|---|---|
| **时序推理** | "第一次"、"之前"、"以来"、"when" | 自动解析查询文本中的日期范围；填充 `date_from`/`date_to` 过滤器；提升较早上下文以获取历史覆盖 | 7 天新鲜度半衰期 |
| **知识更新** | "当前"、"最新"、"新的"、"current" | 强烈提升最新条目；通过时效衰减抑制过时重复 | 30 天半衰期（更强的时效偏好） |
| **多会话** | "所有"、"每个"、"总共"、"all" | 确保结果跨多个会话；从 DB 补充未覆盖会话；按会话去重 | 会话协同检索 min_turns=2 |
| **偏好** | "偏好"、"推荐"、"建议"、"prefer" | 提升用户角色条目 (1.8×)、偏好分类 (2.0×)、隐含指标 (购买/尝试/使用 × 1.3+)、用户+事件交叉 (1.4×) | 多级偏好信号 |
| **聚合** | "多少"、"总共"、"how many" | 自动提升 `top_k`（默认 → 120）以实现穷尽召回；确保跨所有条目的完整枚举 | agg_top_k=120 |

### 会话协同检索 (D1)

当搜索结果属于具有足够上下文（≥2 轮）的会话时，AgentBase 自动**扩展结果集**，拉取同一会话中的相关条目：

- **基于链接的扩展** — 跟随会话提交时存储的 `session_memory_links` 查找协同相关条目
- **轮次感知门控** — 仅对 ≥2 轮的会话激活（可配置），防止单轮会话引入噪声
- **预算感知** — 协同检索的条目遵守 token 预算，不会超过调用方的限制

这解决了"上下文丢失"问题：当检索命中对话中的某条消息时，周围上下文会被自动包含。

### 启发式重排序

在三路 RRF 融合 + NER 提升之后，确定性 5 维重排序器调整分数：

```
final_score = α·rrf_score + β·freshness + γ·confidence + δ·scope_priority + ε·type_match
```

| 维度 | 算法 | 细节 |
|---|---|---|
| **α·RRF 分数** (0.6) | 加权 RRF 输出 | FTS+向量+NER 融合的基础排名 |
| **β·时效性** (0.15) | 指数衰减：`exp(-0.693 · age / half_life)` | 通用 7 天半衰期，知识更新查询 30 天半衰期 |
| **γ·置信度** (0.1) | 来自 ingest 管线的条目级分数 | IDH 硬化的置信度分数 (0.0-1.0) |
| **δ·作用域优先级** (0.1) | 分层：会话(1.0) > 代理(0.8) > 项目(0.6) > 全局(0.4) + 作用域匹配 +0.2 加分 | 更具体的作用域 + 匹配查询作用域 = 更高 |
| **ε·类型匹配** (0.05) | 二元：匹配查询过滤器 1.0，无过滤器 0.5，不匹配 0.0 | 确保类型相关结果优先 |

### 层级递进搜索 (L0 → L1 → L2)

当 `strategy=hierarchical` 时，AgentBase 使用**渐进式精化**方法，在越来越详细的内容层级上搜索：

1. **L0 粗搜索** — 仅搜索 `l0_abstract` 列（短摘要），超取 3× top_k 以获取广泛召回
2. **L1 精搜索** — 在 `l1_overview` 列（中等细节）上重新搜索，收窄至 2× top_k
3. **L2 全文加载** — 仅对最终 top_k 结果加载 `l2_full` 内容

与为所有候选加载全文相比，I/O 和内存使用减少约 60%，同时保持召回质量。

### LLM 增强路径（可选，显式开启）

当配置了 LLM 时，可使用增强检索路径：

1. **意图分解** — LLM 将复杂查询拆分为带分类标签的类型化子查询（memory/resource/skill + profile/preference/entity/event）
2. **逐子查询搜索** — 每个子查询使用各自的类型过滤器独立搜索，结果去重
3. **LLM 重排序** — 使用判断提示对合并结果进行语义重排序，返回 `[index]` 顺序
4. **会话记忆链接** — 通过存储的链接关联实现跨会话协同检索

此路径**默认关闭**，仅在配置了 LLM 且 `strategy=hierarchical` 时激活。

### 完整管线逐步详解

```
1. 查询规范化
   └─ 小写化、去首尾空格、去特殊字符

2. 意图检测（基于规则，零 LLM）
   └─ 检测：时序推理 / 知识更新 / 多会话 / 偏好 / 聚合
   └─ D5：自动解析时间表达式 → 填充 date_from/date_to 过滤器
   └─ D2：聚合检测 → 自动提升 top_k (20→120)

3. 查询分解（零 LLM）
   └─ 通过模式匹配提取关键名词短语
   └─ 双语停用词移除 → 关键词子查询
   └─ D3：提取时序 token（"2022年5月"、"3周前"）
   └─ 按条目 ID 去重子查询结果

4. 三路搜索
   ├─ FTS5 (BM25)：SQLite 内置全文，零延迟
   ├─ sqlite-vec：cosine distance 向量搜索
   └─ 超取 3× top_k 以获取更好的 RRF 召回

5. RRF 融合
   └─ score(d) = Σ weight_i / (k + rank_i(d))
   └─ 默认：FTS=0.4、Vector=0.6、k=60
   └─ 优雅降级：向量不可用 → 纯 FTS + degrade_reason

6. NER 提升（仅对已有结果）
   └─ 通过 spaCy + 正则从查询中提取实体
   └─ 与结果上的 ner_* 标签匹配
   └─ 强：标签匹配 → score *= (1 + 0.3)
   └─ 中：NER 标签 + 内容匹配 → score *= (1 + 0.24)
   └─ 弱：仅内容匹配 → score *= (1 + 0.15)
   └─ 不添加新结果（防止稀释）

7. 查询类型策略
   ├─ 时序：日期过滤器 + 较早上下文提升
   ├─ 知识更新：强时效性 (30天半衰期)
   ├─ 多会话：跨会话补全 + 去重
   ├─ 偏好：用户角色 (1.8×) + 偏好分类 (2.0×) + 隐含信号
   └─ 聚合：top_k 已在步骤 2 提升

8. 会话协同检索 (D1)
   └─ 跟随 session_memory_links（≥2 轮会话）
   └─ 预算感知扩展

9. 启发式重排序（5 维）
   └─ α(0.6)·rrf + β(0.15)·时效性 + γ(0.1)·置信度 + δ(0.1)·作用域 + ε(0.05)·类型

10. LLM 重排序（可选，仅层级策略）
    └─ 判断提示返回 [index] 顺序

11. 加载层级选择
    └─ top_k>20 → L0，budget<1000 → L0，层级 → L1，默认 → L1
    └─ L2 仅对最终结果加载

12. Token 预算裁剪
    └─ 裁剪结果以适应 token_budget

13. 最终 top_k 裁剪 → 返回结果及追踪信息
```

## 架构

```
┌─────────────────────────────────────────────┐
│                  SDK (agentbase)             │
├──────────┬──────────┬───────────┬───────────┤
│   CLI    │   MCP    │    Web    │ Adapters  │
├──────────┴──────────┴───────────┴───────────┤
│              Core Engine                     │
│  ┌─────┐ ┌──────┐ ┌──────┐ ┌────────────┐  │
│  │Store│ │Index │ │Ingest│ │ Retrieval  │  │
│  │SQLite│ │FTS+Vec│ │Pipe │ │ Engine     │  │
│  └─────┘ └──────┘ └──────┘ └────────────┘  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌────────────┐ │
│  │Graph │ │Session│ │Obser│ │ Background │ │
│  │ /NER │ │ Mgmt  │ │vabil│ │   Jobs     │ │
│  └──────┘ └──────┘ └──────┘ └────────────┘ │
└─────────────────────────────────────────────┘
```

## 项目结构

```
agentbase-open/
├── packages/
│   ├── agentbase-core/    # 核心引擎（存储、索引、检索）
│   ├── agentbase-sdk/     # Python SDK + 适配器（LlamaIndex、LangChain）
│   ├── agentbase-cli/     # 命令行接口
│   ├── agentbase-mcp/     # MCP 协议服务器
│   └── agentbase-web/     # Web 仪表盘（FastAPI）
├── tests/                 # 测试套件
├── docs/                  # 文档
├── benchmarks/            # 评测脚本
├── SPEC.md                # 技术规范
└── pyproject.toml         # 工作区配置
```

## 配置

复制示例配置并自定义：

```bash
cp agentbase.yaml.example agentbase.yaml
```

主要配置项：
- **embedding** — 向量嵌入模型（兼容 OpenAI）
- **llm** — 用于摘要、提取和层级生成的 LLM
- **index** — FTS/向量搜索设置和 RRF 融合权重
- **graph** — 知识图谱（实体、关系、遍历）
- **session** — 对话管理和记忆提取
- **tier** — L0/L1/L2 分层内容生成
- **observability** — 追踪、指标和调试

环境变量可覆盖 YAML 配置：`AGENTBASE_<SECTION>__<KEY>`

## 系统要求

- Python >= 3.11
- SQLite 且支持 FTS5
- 可选：`sqlite-vec` 用于向量搜索，`litellm` 用于 LLM 功能

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
