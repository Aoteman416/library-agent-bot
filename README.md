# 图书馆智能客服（Library Agent Bot）

一个基于 **FastAPI + RAG** 的图书馆智能客服系统，支持检索问答、闲聊、任务型业务（座位预约 / 图书续借 / 借阅查询）与多层级安全守卫。内置混合检索、对话状态跟踪（DST）、意图路由、熔断降级与评测体系。

> 本仓库只包含可公开的项目源码与资源；私人的学习 / 面试准备文档（`docs/`、`interview/` 等）已刻意排除，不会上传。

## 特性

- **四通路意图路由**：检索问答 / 闲聊 / 任务型 / 兜底澄清，按置信度校准分流。
- **混合检索**：BM25 关键词召回 + BGE-M3 稠密向量召回，RRF 融合 + Cross-Encoder 重排（Top-3）。
- **自研 DST 状态机**：多轮对话上下文跟踪与状态管理。
- **L1–L4 四层安全守卫**：输入净化 / 语义安全分类（BGE-M3 + FAISS）/ 轻量分类器 / 规则兜底。
- **任务型 function calling**：座位预约、图书续借、借阅与罚金查询（Mock ILS，内存落库）。
- **工程化能力**：JWT 鉴权 + 刷新令牌、限流、Redis 语义缓存、熔断降级（M2.7）、统一错误码。
- **评测体系**：意图 / 检索 / 安全 / 守卫 多维评测集与脚本。
- **离线可用**：未配置 `LLM_API_KEY` 时自动降级为模板应答；embedding 缺失时回退稀疏检索，开箱即用。

## 技术栈

- **后端**：FastAPI（SSE 流式）+ Uvicorn
- **检索**：BM25（`rank_bm25`）+ BGE-M3（`sentence-transformers`）+ Chroma 向量库 + Cross-Encoder 重排
- **模型接入**：兼容 OpenAI `/chat/completions` 的多 Provider（SiliconFlow / DashScope / 千帆 / 豆包 / One-API / OpenAI）
- **安全**：BGE-M3 + FAISS 语义分类 + scikit-learn 轻量分类器（`l3_clf.joblib`）
- **存储**：SQLAlchemy（MySQL / SQLite 演示）+ Redis（可选）
- **前端**：原生 HTML / CSS / JS 静态页面

## 项目结构

```
.
├── app/                  # FastAPI 应用
│   ├── api/v1/           # 路由：chat / auth / book / user / seat / admin
│   ├── core/             # DST / 路由 / 守卫(guard) / 熔断 / 限流 / 鉴权 ...
│   ├── rag/              # 混合检索 / 重排 / 向量库 / BM25 / 分块
│   ├── services/         # 业务服务：chat / rag / seat / user / book / admin
│   ├── models/           # 数据库与 Pydantic Schema
│   ├── config.py         # 配置（读取 .env）
│   └── main.py           # 应用入口 / lifespan 初始化
├── web/                  # 前端静态资源（index.html / css / js）
├── data/                 # 知识库、守卫资产、评测集
│   ├── knowledge/        # RAG 知识文档（MD）
│   ├── guard/            # 安全分类器模型与攻击模板
│   └── eval/             # 评测数据集与报告
├── eval/                 # 评测脚本（意图 / 检索 / 安全 / 守卫）
├── scripts/              # 构建与数据处理脚本
├── tests/                # 单元测试
├── run.py                # 一键启动
├── Dockerfile / docker-compose.yml
└── requirements*.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 接入 BGE-M3 + Chroma + 重排（推荐，检索效果最佳）
pip install -r requirements-rag.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填 LLM_PROVIDER 与 LLM_API_KEY
# （也可手动填 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY）
```

### 3. 启动

```bash
python run.py
# 或：uvicorn app.main:app --reload --port 8000
```

- 服务地址：<http://localhost:8000>
- API 文档：<http://localhost:8000/docs>
- 前端界面：<http://localhost:8000/>

> 未配置 `LLM_API_KEY` 时，系统以**离线模板模式**运行（检索问答仍可用，闲聊走兜底），开箱即用。

## Docker

```bash
docker compose up --build
```

## API 概览（前缀 `/api/v1`）

| 分组 | 端点 | 说明 |
| --- | --- | --- |
| chat | `POST /chat/stream`、`POST /chat` | 对话（SSE 流式 / 普通） |
| auth | `POST /auth/refresh`、`GET /auth/me` | 鉴权与令牌刷新 |
| book | `GET /book/search`、`GET /book/{id}` | 图书检索 |
| user | `GET /user/borrows`、`POST /user/renew`、`GET /user/fines` | 借阅 / 续借 / 罚金 |
| seat | `GET /seat/availability`、`POST /seat/reserve`、`POST /seat/cancel` | 座位预约 |
| admin | `GET /admin/stats/*`、`POST /admin/kb/*` | 管理后台与知识库回流 |

## 评测

```bash
python -m eval.run_eval          # 跑全部评测
# 或单独运行 eval/run_*_eval.py（意图 / 检索 / 安全 / 守卫）
```

评测集位于 `data/eval/`，覆盖意图分类、检索召回、安全攻击与守卫拦截，报告输出到 `data/eval/report.md`。

## 已知限制（演示级 → 生产级差距）

- 真实借还 / 支付 / 工单系统为 **Mock**（内存落库），未对接真实 ILS（OPAC）。
- 评测集规模有限（约 200+ 条），自标自评，作为**回归闸门**而非泛化证明。
- 间接注入防护仅扫描用户输入，未扫描检索文档。
- 多实例部署下会话 / 缓存为进程内存（非 Redis 集群）。

## License

[MIT](./LICENSE)
