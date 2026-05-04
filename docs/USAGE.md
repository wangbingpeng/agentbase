# AgentBase 用户手册

[English](USAGE_EN.md)

> AgentBase — AI 智能体的上下文数据库
> Context DB = Memory + Resource + Skill + Temporal + Session + Observability

---

## 目录

1. [快速开始](#1-快速开始)
2. [安装](#2-安装)
3. [核心概念](#3-核心概念)
4. [配置参考](#4-配置参考)
5. [Python SDK](#5-python-sdk)
6. [CLI 命令行工具](#6-cli-命令行工具)
7. [MCP 协议集成](#7-mcp协议集成)
8. [框架适配器](#8-框架适配器)
9. [搜索与检索](#9-搜索与检索)
10. [知识图谱](#10-知识图谱)
11. [会话管理](#11-会话管理)
12. [可观测性](#12-可观测性)
13. [维护操作](#13-维护操作)
14. [生产部署](#14-生产部署)
15. [错误处理](#15-错误处理)
16. [性能调优](#16-性能调优)

---

## 1. 快速开始

```python
import asyncio
from agentbase import AgentBase

async def main():
    async with AgentBase(path="./my_agent.db") as db:
        # 写入记忆
        entry = await db.add_memory(
            "用户偏好使用 Python 3.12 和 VS Code",
            category="preference",
            tags=["python", "ide"],
            confidence=0.95,
        )

        # 搜索上下文
        results = await db.find("Python 偏好", top_k=5)
        for r in results:
            print(f"[{r.score:.2f}] {r.entry.l2_full}")

        # 创建会话
        session = await db.create_session(agent_id="my-agent")
        await db.add_message(session.id, "user", "帮我写一个 Python 脚本")
        await db.add_message(session.id, "assistant", "好的，我来帮你写...")

        # 提交会话，压缩并提取记忆
        memories = await db.commit_session(session.id)

asyncio.run(main())
```

---

## 2. 安装

### 2.1 从源码安装（推荐）

```bash
git clone <repo-url> agentbase
cd agentbase

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2.2 包结构

| 包名 | 说明 |
|------|------|
| `agentbase-core` | 核心引擎、存储、索引、检索 |
| `agentbase-sdk` | Python SDK 公共接口 |
| `agentbase-cli` | 命令行工具 |
| `agentbase-mcp` | MCP 协议服务 |
| `agentbase-web` | Web 仪表盘与可观测性界面 |

### 2.3 依赖

以下依赖随 `agentbase-core` 自动安装：

| 依赖 | 用途 |
|------|------|
| `sqlite-vec` | 向量搜索引擎（通过 `vector_enabled` 控制开关） |
| `pyyaml` | YAML 配置文件读写 |
| `pydantic` / `pydantic-settings` | 数据模型与配置 |
| `aiosqlite` | 异步 SQLite 驱动 |
| `python-ulid` | ULID ID 生成 |

> **零外部数据库依赖**：所有功能基于 SQLite 运行，无需 Docker、无需云服务。

---

## 3. 核心概念

### 3.1 三种上下文类型

| 类型 | 用途 | 专属字段 |
|------|------|----------|
| **Memory** | 记忆、偏好、经验 | `memory_category` |
| **Resource** | 资源、文档、链接 | `resource_url`, `resource_format`, `resource_size` |
| **Skill** | 技能、工具定义 | `skill_tool_name`, `skill_api_spec` |

> 字段约束：每种类型只能使用对应字段，跨类型字段在写入时会被校验拒绝。

### 3.2 四层作用域

| 作用域 | 可见性 | owner_id |
|--------|--------|----------|
| `global` | 所有智能体可见 | 必须为 None |
| `project` | 同项目智能体可见 | 必须设置（项目ID） |
| `agent` | 仅特定智能体可见 | 必须设置（智能体ID） |
| `session` | 仅特定会话可见 | 必须设置（会话ID） |

### 3.3 三层内容

| 层级 | 长度 | 用途 |
|------|------|------|
| L0 Abstract | 20-50 字符 | 快速过滤、标题 |
| L1 Overview | 100-300 字符 | 摘要、概览 |
| L2 Full | 完整内容 | 详细信息 |

### 3.4 六种记忆类别

`profile` | `preference` | `entity` | `event` | `case` | `pattern`

### 3.5 特性标志

| 特性 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| 向量检索 | `config.index.vector_enabled` | `True` | embedder 不可用时自动降级为纯 FTS |
| 知识图谱 | `config.graph.enabled` | `True` | CRUD 始终可用；LLM 提取自动降级 |
| 会话管理 | `config.session.enabled` | `True` | 提交始终可用；LLM 提取自动降级 |
| 可观测性 | `config.observability.enabled` | `False` | 调试/监控时开启 |

> AgentBase 遵循**优雅降级**原则：LLM 不可用时，功能降级为本地规则而非崩溃。

---

## 4. 配置参考

### 4.1 YAML 配置文件

创建 `agentbase.yaml`：

```yaml
data_dir: ~/.agentbase
db_filename: agentbase.db

embedding:
  model: text-embedding-3-small
  dimensions: 1536
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  max_concurrent: 10

llm:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1
  max_tokens: 1024

index:
  vector_enabled: true          # 启用 FTS + 向量混合检索
  tokenizer: auto               # 自动检测语言 | jieba | char
  fts_weight: 0.5               # FTS BM25 权重
  vec_weight: 0.5               # 向量余弦权重
  rrf_k: 60                     # RRF 平滑常数
  dedup_threshold: 0.92         # 去重相似度阈值

graph:
  enabled: true
  max_traversal_depth: 4
  max_entities: 10000
  max_relations: 50000
  extract_on_ingest: false      # 摄取时自动提取实体

session:
  enabled: true
  keep_recent_turns: 6
  auto_commit: false
  extract_memories: true
  extract_on_ingest: false      # ingest_direct 时自动创建会话

ingest:
  session_summary: true         # LLM 会话摘要（降级为截断）
  fact_extraction: true         # LLM 事实提取（降级为本地正则）
  ner_extraction: true          # NER 实体提取（双语）

retrieval:
  default_top_k: 20
  default_token_budget: 24000
  freshness_half_life_days: 7.0
  knowledge_update_half_life_days: 14.0
  query_decomposition: true     # 本地规则查询分解
  ner_boost: true               # NER 感知查询扩展 + 结果提升
  ner_weight: 0.3               # NER 信号权重（三路融合）
  session_co_retrieval: true    # 会话协同检索 (D1)
  co_retrieve_min_turns: 2      # 协同检索激活的最少轮次
  agg_top_k: 120                # 聚合查询 top_k
  agg_detection: true           # 自动检测聚合查询 + 提升 top_k

tier:
  enabled: true
  async_generation: true
  max_concurrent: 5
  fallback_to_truncation: true  # 无 LLM 时截断生成 L0/L1

observability:
  enabled: false
  persist_traces: true
  trace_sample_rate: 1.0
  max_trace_age_days: 30
```

### 4.2 从 YAML 加载

```python
from agentbase_core.models.config import AgentBaseConfig

# 从文件加载（环境变量优先级高于 YAML）
config = AgentBaseConfig.from_yaml("agentbase.yaml")

# 保存配置到文件
config.to_yaml("agentbase.yaml")
```

### 4.3 环境变量

所有配置项均支持 `AGENTBASE_` 前缀环境变量，嵌套使用 `__` 分隔：

```bash
# 基础配置
export AGENTBASE_DATA_DIR=/data/agentbase
export AGENTBASE_DB_FILENAME=agentbase.db

# 嵌套配置
export AGENTBASE_EMBEDDING__MODEL=text-embedding-3-small
export AGENTBASE_EMBEDDING__DIMENSIONS=1536
export AGENTBASE_EMBEDDING__API_KEY=sk-xxx
export AGENTBASE_LLM__MODEL=gpt-4o-mini
export AGENTBASE_LLM__API_KEY=sk-xxx

# 特性开关
export AGENTBASE_INDEX__VECTOR_ENABLED=true
export AGENTBASE_GRAPH__ENABLED=true
export AGENTBASE_SESSION__ENABLED=true
export AGENTBASE_OBSERVABILITY__ENABLED=true

# 检索调优
export AGENTBASE_RETRIEVAL__NER_BOOST=true
export AGENTBASE_RETRIEVAL__NER_WEIGHT=0.3
export AGENTBASE_RETRIEVAL__SESSION_CO_RETRIEVAL=true
export AGENTBASE_RETRIEVAL__AGG_DETECTION=true
```

### 4.4 配置优先级

**环境变量 > YAML 文件 > 代码默认值**

### 4.5 完整配置表

| 分组 | 键名 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | `data_dir` | Path | `~/.agentbase` | 数据存储目录 |
| — | `db_filename` | str | `agentbase.db` | SQLite 数据库文件名 |
| embedding | `model` | str | `text-embedding-3-small` | 嵌入模型名称 |
| embedding | `dimensions` | int | `1536` | 向量维度 |
| embedding | `api_base` | str? | None | API 地址 |
| embedding | `api_key` | str? | None | API 密钥 |
| embedding | `max_concurrent` | int | `10` | 最大并发嵌入请求 |
| llm | `model` | str | `gpt-4o-mini` | LLM 模型名称 |
| llm | `api_base` | str? | None | API 地址 |
| llm | `api_key` | str? | None | API 密钥 |
| llm | `temperature` | float | `0.1` | 生成温度 |
| llm | `max_tokens` | int | `1024` | 单次最大生成 token |
| index | `vector_enabled` | bool | `true` | 启用向量检索 |
| index | `tokenizer` | str | `auto` | 分词器：auto/jieba/char |
| index | `fts_weight` | float | `0.5` | FTS BM25 权重 |
| index | `vec_weight` | float | `0.5` | 向量余弦权重 |
| index | `rrf_k` | int | `60` | RRF 平滑常数 |
| index | `dedup_threshold` | float | `0.92` | 去重相似度阈值 |
| graph | `enabled` | bool | `true` | 启用知识图谱 |
| graph | `max_traversal_depth` | int | `4` | 图遍历最大深度 |
| graph | `max_entities` | int | `10000` | 实体数量上限 |
| graph | `max_relations` | int | `50000` | 关系数量上限 |
| graph | `extract_on_ingest` | bool | `false` | 摄取时自动提取 |
| session | `enabled` | bool | `true` | 启用会话管理 |
| session | `keep_recent_turns` | int | `6` | 保留最近轮次 |
| session | `auto_commit` | bool | `false` | 自动提交会话 |
| session | `extract_memories` | bool | `true` | 提交时提取记忆 |
| session | `extract_on_ingest` | bool | `false` | ingest_direct 时自动创建会话 |
| ingest | `session_summary` | bool | `true` | 会话摘要生成 |
| ingest | `fact_extraction` | bool | `true` | 事实提取 |
| ingest | `ner_extraction` | bool | `true` | NER 实体提取 |
| ingest | `extract_on_direct_ingest` | bool | `false` | ingest_direct 时提取 |
| retrieval | `default_top_k` | int | `20` | 默认结果数量 |
| retrieval | `default_token_budget` | int | `24000` | 默认 token 预算 |
| retrieval | `freshness_half_life_days` | float | `7.0` | 新鲜度半衰期 |
| retrieval | `knowledge_update_half_life_days` | float | `14.0` | 知识更新半衰期 |
| retrieval | `query_decomposition` | bool | `true` | 查询分解 |
| retrieval | `ner_boost` | bool | `true` | NER 实体提升 |
| retrieval | `ner_weight` | float | `0.3` | NER 提升权重 |
| retrieval | `session_co_retrieval` | bool | `true` | 会话协同检索 |
| retrieval | `co_retrieve_min_turns` | int | `2` | 协同检索最少轮次 |
| retrieval | `agg_top_k` | int | `120` | 聚合查询 top_k |
| retrieval | `agg_detection` | bool | `true` | 聚合查询检测 |
| tier | `enabled` | bool | `true` | 启用 L0/L1 生成 |
| tier | `async_generation` | bool | `true` | 异步层级生成 |
| tier | `max_concurrent` | int | `5` | 最大并发生成 |
| tier | `fallback_to_truncation` | bool | `true` | 截断降级 |
| observability | `enabled` | bool | `false` | 启用可观测性 |
| observability | `persist_traces` | bool | `true` | 持久化追踪 |
| observability | `trace_sample_rate` | float | `1.0` | 追踪采样率 |
| observability | `max_trace_age_days` | int | `30` | 追踪最大保留天数 |

---

## 5. Python SDK

### 5.1 初始化

```python
from agentbase import AgentBase

# 方式1：指定数据库路径
db = AgentBase(path="./my_agent.db")
await db.initialize()

# 方式2：使用配置对象
from agentbase_core.models.config import AgentBaseConfig, GraphConfig, SessionConfig

config = AgentBaseConfig(
    data_dir=Path("/data/agentbase"),
    graph=GraphConfig(enabled=True),
    session=SessionConfig(enabled=True),
)
db = AgentBase(config=config)
await db.initialize()

# 方式3：上下文管理器（自动初始化和关闭）
async with AgentBase(path="./my_agent.db") as db:
    # 使用 db ...
    pass
```

### 5.2 写入操作

```python
# 添加记忆
entry = await db.add_memory(
    content="用户偏好暗色主题",
    category="preference",        # profile/preference/entity/event/case/pattern
    tags=["ui", "theme"],
    confidence=0.9,
    scope="agent",                # global/agent/project/session
    owner_id="agent-001",
)

# 添加资源
entry = await db.add_resource(
    url="https://docs.python.org/3/",
    content="Python 官方文档",
    format="html",
    tags=["python", "docs"],
    confidence=1.0,
    scope="global",
)

# 添加技能
entry = await db.add_skill(
    tool_name="web_search",
    description="搜索互联网获取信息",
    api_spec={"endpoint": "/search", "method": "GET"},
    tags=["search", "web"],
    confidence=1.0,
)
```

### 5.3 读取操作

```python
# 按 ID 获取
entry = await db.get(entry_id="01HX...")

# 简单搜索
results = await db.find("Python 文档", top_k=5)

# 高级搜索
from agentbase import SearchQuery, SearchStrategy, ContextType, EntryStatus

query = SearchQuery(
    text="Python 教程",
    top_k=10,
    strategy="hybrid",              # fts/vector/hybrid/hierarchical
    context_type=ContextType.RESOURCE,
    scope="global",
    tags=["python"],
    min_confidence=0.7,
    token_budget=4000,
    load_level="auto",              # auto/l0/l1/l2
    include_trace=True,
    include_statuses=[EntryStatus.ACTIVE],
)
results = await db.search(query)

for r in results:
    print(f"Score: {r.score:.3f} | Stage: {r.ranking_stage} | Match: {r.matched_by}")
    print(f"  Degrade: {r.degrade_reason}")  # None/vec_unavailable/embedding_failed
    print(f"  Content: {r.entry.l2_full[:100]}")
```

### 5.4 删除操作

```python
# 软删除（标记为 deleted，仍可查询）
await db.delete(entry_id="01HX...")

# 硬删除（物理删除，不可恢复）
await db.purge(entry_id="01HX...")
```

### 5.5 列表与统计

```python
# 列表查询
entries = await db.list_entries(
    scope="agent",
    context_type="memory",
    limit=50,
    offset=0,
)

# 统计数量
count = await db.count(scope="agent", context_type="memory")
```

### 5.6 文本摄取

```python
# 通过 LLM 从原始文本中提取结构化记忆
entries = await db.ingest_text(
    text="用户在项目中使用了 FastAPI 框架...",
    context_type="memory",
    scope="project",
    owner_id="project-001",
    tags=["framework"],
)
```

---

## 6. CLI 命令行工具

### 6.1 初始化

```bash
agentbase init --path ./my_agent.db
```

### 6.2 添加条目

```bash
# 添加记忆
agentbase add "用户偏好 Python 3.12" --type memory --category preference --tags "python,version" --scope global

# 添加资源
agentbase add "Python 官方文档" --type resource --tags "docs,python"

# 添加技能
agentbase add "Web 搜索工具" --type skill --tags "search"
```

### 6.3 搜索

```bash
agentbase find "Python 偏好" --top-k 5 --type memory
```

### 6.4 获取条目

```bash
agentbase get <entry_id>
```

### 6.5 删除条目

```bash
agentbase delete <entry_id>
```

### 6.6 会话管理

```bash
# 创建会话
agentbase session create --agent-id my-agent

# 添加消息
agentbase session add-message <session_id> --role user --content "你好"

# 查看会话
agentbase session show <session_id>

# 提交会话
agentbase session commit <session_id> --mode full
```

### 6.7 实体操作

```bash
# 添加实体
agentbase entity add "Python" --type concept --description "编程语言"

# 查找实体
agentbase entity find "Python"

# 添加关系
agentbase entity relate <source_id> <target_id> --predicate "uses"

# 图遍历
agentbase entity traverse "Python" --depth 2
```

### 6.8 统计

```bash
agentbase stats
```

### 6.9 维护

```bash
# 重建索引
agentbase reindex

# 清理数据
agentbase cleanup --traces-older-than 30 --deleted-older-than 7

# 调试：解释查询
agentbase debug explain "Python 偏好"

# 调试：查看检索追踪
agentbase debug trace <trace_id>
```

---

## 7. MCP 协议集成

### 7.1 可用工具

| 工具名 | 说明 | 必需参数 |
|--------|------|----------|
| `add_memory` | 添加记忆条目 | `content` |
| `add_resource` | 添加资源条目 | `content` |
| `add_skill` | 添加技能条目 | `tool_name` |
| `find_context` | 搜索上下文 | `query` |
| `get_context` | 获取条目 | `entry_id` |
| `delete_context` | 删除条目 | `entry_id` |
| `add_entity` | 添加实体 | `name` |
| `find_entities` | 查找实体 | `name` |
| `add_relation` | 添加关系 | `source_id`, `target_id`, `predicate` |
| `graph_traverse` | 图遍历 | `entity_name` |
| `create_session` | 创建会话 | (无必需) |
| `add_message` | 添加消息 | `session_id`, `role`, `content` |
| `commit_session` | 提交会话 | `session_id` |
| `get_stats` | 获取统计 | (无参数) |

### 7.2 在 Claude Desktop 中使用

在 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "agentbase": {
      "command": "python",
      "args": ["-m", "agentbase_mcp"],
      "env": {
        "AGENTBASE_DB_FILENAME": "agentbase.db",
        "AGENTBASE_GRAPH__ENABLED": "true",
        "AGENTBASE_SESSION__ENABLED": "true"
      }
    }
  }
}
```

---

## 8. 框架适配器

AgentBase 提供 5 个适配器，可作为主流 AI 框架的统一记忆后端：

### 8.1 Mem0Adapter — 一行替换 Mem0

```python
from agentbase import Mem0Adapter

# 替换：from mem0 import Memory; m = Memory()
memory = Mem0Adapter(db_path="./agentbase.db")

await memory.add("我偏好暗色模式", user_id="user-1")
results = await memory.search("主题偏好", user_id="user-1")
all_memories = await memory.get_all(user_id="user-1")
```

### 8.2 LangChainMemoryAdapter — 鸭子类型兼容 BaseChatMemory

```python
from agentbase import LangChainMemoryAdapter

memory = LangChainMemoryAdapter(db_path="./agentbase.db", session_id="session-1")

await memory.save_context(
    inputs={"input": "什么是 Python？"},
    outputs={"output": "Python 是一种编程语言。"}
)
result = await memory.load_memory_variables({"input": "Python"})
```

### 8.3 AgentBaseChatStore — LlamaIndex BaseChatStore

```python
from agentbase import AgentBaseChatStore

store = AgentBaseChatStore(db_path="./agentbase.db")
from llama_index.core.llms import ChatMessage
store.set_messages("chat-key", [ChatMessage(role="user", content="你好")])
messages = store.get_messages("chat-key")
```

### 8.4 OpenAIAssistantAdapter — 映射 Thread→Session

```python
from agentbase import OpenAIAssistantAdapter

adapter = OpenAIAssistantAdapter(db_path="./agentbase.db")
thread = await adapter.create_thread(metadata={"assistant_id": "asst-1"})
await adapter.add_message(thread.id, role="user", content="帮我写代码")
messages = await adapter.list_messages(thread.id)
```

### 8.5 MinimalAdapter — 三方法 API

```python
from agentbase import MinimalAdapter

memory = MinimalAdapter(db_path="./agentbase.db")

await memory.remember("用户偏好 Python 3.12")
results = await memory.recall("Python 偏好")
await memory.forget(entry_id="01HX...")
```

---

## 9. 搜索与检索

### 9.1 搜索策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `hybrid` | FTS + 向量 + NER + RRF 融合 | 通用搜索（默认） |
| `fts` | 仅全文搜索 | 关键词精确匹配 |
| `vector` | 仅向量搜索 | 语义相似匹配 |
| `hierarchical` | L0→L1→L2 渐进式 | 大规模数据集 |

### 9.2 检索流水线

```
查询文本 → 查询规范化 → 意图检测 → 策略路由
→ 三路搜索 (FTS + 向量 + NER) → RRF 融合 → NER 提升
→ 启发式重排 → LLM 重排(可选) → 加载层级 → 返回结果
```

### 9.3 三路混合搜索 (FTS + 向量 + NER)

默认情况下，AgentBase 使用**三路混合搜索**，组合了：

1. **FTS5 (BM25)**：通过 SQLite FTS5 进行全文关键词匹配
2. **sqlite-vec (cosine)**：向量相似度搜索
3. **NER 实体提升**：对实体标签结果进行分数放大

三项默认全部开启——零配置即可使用：

```yaml
retrieval:
  ner_boost: true     # 默认：true
  ner_weight: 0.3     # 默认：0.3
```

关闭 NER 提升：

```yaml
retrieval:
  ner_boost: false
```

### 9.4 RRF 融合参数

```yaml
index:
  fts_weight: 0.5     # FTS BM25 权重
  vec_weight: 0.5     # 向量相似度权重
  rrf_k: 60           # RRF 平滑常数
```

### 9.5 启发式重排序权重

5 维启发式重排序公式：

```
最终分数 = α(0.6)·rrf + β(0.15)·freshness + γ(0.1)·confidence + δ(0.1)·scope + ε(0.05)·type
```

| 权重 | 维度 | 说明 |
|------|------|------|
| α (0.6) | RRF 分数 | 原始混合搜索分数 |
| β (0.15) | 新鲜度 | 时间衰减（7天半衰期） |
| γ (0.1) | 置信度 | 原始条目置信度 |
| δ (0.1) | 作用域优先级 | session > agent > project > global |
| ε (0.05) | 类型匹配 | 类型匹配奖励 |

### 9.6 查询类型感知检索

AgentBase 自动检测 5 种查询类型并应用专门策略：

| 查询类型 | 自动检测 | 专门策略 |
|----------|----------|----------|
| `temporal-reasoning` | 日期表达式 | 自动填充 `date_from`/`date_to` 过滤器 |
| `knowledge-update` | 时效关键词 | 双半衰期 (7d / 14d)，强时效偏好 |
| `multi-session` | 跨会话关键词 | 会话补全 + 去重 |
| `preference` | 偏好指示词 | 用户角色 (1.8×) + 偏好分类 (2.0×) 提升 |
| `aggregation` | "多少"、"总计" | 自动提升 top_k 20→120 |

### 9.7 load_level 自动规则

| 条件 | 层级 |
|------|------|
| top_k > 20 | L0 |
| token_budget < 1000 | L0 |
| strategy = hierarchical | L1 |
| resource + l2_full > 500 字符 | L1 |
| memory + l2_full < 200 字符 | L2 |
| 默认 | L1 |

### 9.8 向量检索控制

```yaml
index:
  vector_enabled: true   # 启用向量检索
```

或环境变量：

```bash
export AGENTBASE_INDEX__VECTOR_ENABLED=true
```

| 状态 | 搜索行为 | 结果标记 |
|------|----------|----------|
| `vector_enabled=false` | 仅 FTS BM25 关键词匹配 | `degrade_reason = "vec_unavailable"` |
| `vector_enabled=true` | FTS + 向量余弦 + RRF 融合 | `degrade_reason = None` |

---

## 10. 知识图谱

> 需要启用：`config.graph.enabled = True`（默认）

### 10.1 核心模型

**Entity（实体）**

| 字段 | 类型 | 说明 |
|------|------|------|
| name | str | 实体名称 |
| entity_type | str | person/project/concept/tool/event/organization |
| description | str | 描述 |
| properties | dict | 扩展属性 |

**Relation（关系）**

| 字段 | 类型 | 说明 |
|------|------|------|
| source_id | str | 源实体 ID |
| target_id | str | 目标实体 ID |
| predicate | str | 关系谓词 |
| confidence | float | 置信度 |
| valid_until | datetime? | 失效时间 |

### 10.2 SDK 操作

```python
entity = await db.add_entity("Python", entity_type="concept", description="编程语言")
await db.add_alias(entity.id, "Python3")
entities = await db.find_entities("Python")
await db.add_relation(source_id=ent1.id, target_id=ent2.id, predicate="uses")
relations = await db.get_current_relations(entity.id)
paths = await db.graph_traversal("Python", depth=2)
await db.add_fact(entity.id, fact="Python 3.12 发布于 2023 年")
facts = await db.get_current_facts(entity.id)
```

---

## 11. 会话管理

> 需要启用：`config.session.enabled = True`（默认）

### 11.1 会话生命周期

```
创建 → 添加消息 → 提交(commit) → 归档
```

### 11.2 提交模式

| 模式 | 说明 |
|------|------|
| `full` | 压缩 + 记忆提取（默认） |
| `archive_only` | 仅压缩归档 |
| `extract_only` | 仅记忆提取 |

### 11.3 SDK 操作

```python
session = await db.create_session(agent_id="my-agent", project="project-001")
await db.add_message(session.id, "user", "帮我写一个爬虫")
await db.add_message(session.id, "assistant", "好的，我来写一个...")
session = await db.get_session(session.id, load_messages=True)
memories = await db.commit_session(session.id, mode="full")
```

### 11.4 会话压缩

- 保留最近 N 轮对话（`keep_recent_turns=6`）
- 较早对话归档为 L0/L1 摘要
- 需要 LLM 生成摘要，无 LLM 时使用截断降级

### 11.5 记忆提取

提交时自动从对话中提取 6 类记忆：

profile / preference / entity / event / case / pattern

每条记忆包含 `category`、`content`、`tags`、`confidence`，置信度低于 0.5 的会被过滤。

---

## 12. 可观测性

> 需要启用：`config.observability.enabled = True`

### 12.1 三个组件

| 组件 | 说明 |
|------|------|
| TraceCollector | 检索追踪采集（可配置采样率） |
| ContextMetrics | 质量指标（查询数、延迟、P50 等） |
| DebugService | 调试工具（查询解释、差异对比等） |

### 12.2 SDK 操作

```python
metrics = await db.get_metrics()
explanation = await db.explain_query("Python 教程")
```

### 12.3 DebugService 方法

```python
from agentbase_core.observability.observability_service import DebugService

svc = DebugService(pool)
trace = await svc.get_trace(trace_id="...")
traces = await svc.list_recent_traces(limit=10)
explanation = await svc.explain_query("Python 教程")
diff = await svc.diff_contexts(id1="...", id2="...")
session_traces = await svc.trace_session(session_id="...")
graph = await svc.entity_graph("Python", depth=2)
```

### 12.4 Web 仪表盘

AgentBase 内置 Web 仪表盘，提供可视化可观测性：

- **时间线**：所有上下文条目的时间顺序视图
- **热力图**：条目密度时间分布
- **分类旭日图**：层级分类分布
- **新鲜度分布**：条目年龄分布
- **标签云**：最频繁标签
- **活动流**：实时条目变更

---

## 13. 维护操作

### 13.1 重建索引

```python
result = await engine.reindex()
```

### 13.2 数据清理

```python
result = await engine.cleanup(
    traces_older_than_days=30,
    deleted_older_than_days=7,
    failed_jobs_older_than_days=14,
)
```

### 13.3 VACUUM

```python
await engine.vacuum()
```

### 13.4 后台任务管理

```python
runner = engine.job_runner
jobs = await runner.list_jobs()
await runner.retry_failed()
await runner.resume_pending()
```

---

## 14. 生产部署

### 14.1 推荐配置

```yaml
data_dir: /data/agentbase
db_filename: agentbase.db

embedding:
  model: text-embedding-3-small
  dimensions: 1536
  api_key: ${OPENAI_API_KEY}
  max_concurrent: 10

llm:
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1
  max_tokens: 1024

index:
  vector_enabled: true
  fts_weight: 0.5
  vec_weight: 0.5
  rrf_k: 60
  dedup_threshold: 0.92

graph:
  enabled: true
  max_traversal_depth: 4

session:
  enabled: true
  keep_recent_turns: 6

retrieval:
  ner_boost: true
  ner_weight: 0.3
  session_co_retrieval: true
  agg_detection: true

observability:
  enabled: true
  trace_sample_rate: 0.1    # 生产环境降低采样率
  max_trace_age_days: 7     # 缩短保留期
```

### 14.2 SQLite 调优

AgentBase 默认使用 WAL 模式。高并发场景下，ConnectionPool 自动管理读写连接。

### 14.3 定时维护

```bash
# 每天清理
0 3 * * * agentbase cleanup --traces-older-than 7 --deleted-older-than 7 --path /data/agentbase/agentbase.db

# 每周重建索引
0 4 * * 0 agentbase reindex --path /data/agentbase/agentbase.db
```

### 14.4 数据库备份

```bash
sqlite3 /data/agentbase/agentbase.db ".backup /backup/agentbase_$(date +%Y%m%d).db"
```

### 14.5 自定义 LLM/Embedder

```python
from agentbase_core.llm.base import AbstractLLM
from agentbase_core.embedding.base import AbstractEmbedder

class MyLLM(AbstractLLM):
    async def complete(self, prompt: str, **kwargs) -> str: ...
    async def complete_json(self, prompt: str, **kwargs) -> dict | list: ...

class MyEmbedder(AbstractEmbedder):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]: ...

engine = AgentBaseEngine(config=config, llm=MyLLM(), embedder=MyEmbedder())
```

### 14.6 多智能体架构

```python
# 全局共享知识
await db.add_memory("项目使用 FastAPI", scope="global")

# 项目级知识
await db.add_memory("API 前缀为 /api/v2", scope="project", owner_id="project-001")

# 智能体私有知识
await db.add_memory("我的角色是代码审查", scope="agent", owner_id="reviewer-agent")

# 搜索时自动按作用域过滤
results = await db.find("API 配置", scope="agent", owner_id="reviewer-agent")
```

---

## 15. 错误处理

### 15.1 异常层级

```
AgentBaseError
+-- StorageError        # SQLite 操作失败
+-- IndexOpError        # 索引操作失败（FTS5/sqlite-vec）
+-- EmbeddingError      # 嵌入生成失败
+-- LLMError            # LLM 调用失败
+-- GraphError          # 图操作失败
+-- SessionError        # 会话操作失败
+-- ConfigError         # 配置错误（特性未启用等）
+-- ConflictError       # 事实冲突需人工介入
+-- BackgroundJobError  # 后台任务执行失败
+-- ValidationError     # 输入校验错误
```

### 15.2 常见错误处理

```python
from agentbase_core.exceptions import AgentBaseError, StorageError, ConfigError, ValidationError

try:
    result = await db.find("query")
except ConfigError as e:
    print(f"需要启用特性: {e}")
except ValidationError as e:
    print(f"校验错误: {e}")
except StorageError as e:
    print(f"存储错误: {e}")
except AgentBaseError as e:
    print(f"错误: {e}")
```

---

## 16. 性能调优

### 16.1 索引权重调整

```yaml
index:
  fts_weight: 0.3    # 降低 FTS 权重（关键词匹配不重要的场景）
  vec_weight: 0.7    # 提高向量权重（语义搜索重要场景）
  rrf_k: 60          # 默认值，一般不需要调整
```

### 16.2 去重阈值

```yaml
index:
  dedup_threshold: 0.92   # 相似度 > 0.92 视为重复
                            # 降低 = 更严格去重
                            # 升高 = 更宽松去重
```

### 16.3 层级生成

```yaml
tier:
  enabled: true
  async_generation: true
  max_concurrent: 5
  fallback_to_truncation: true
```

### 16.4 会话保留轮数

```yaml
session:
  keep_recent_turns: 6    # 保留最近 6 轮对话
```

### 16.5 可观测性采样

```yaml
observability:
  trace_sample_rate: 0.1   # 生产环境：0.01-0.1，开发/调试：1.0
```

### 16.6 Embedding 缓存

AgentBase 自动缓存已计算的嵌入向量（`embedding_cache` 表），相同内容不会重复计算。

---

> **AgentBase** — 让 AI 智能体拥有持久化、可检索、可演进的上下文记忆。
