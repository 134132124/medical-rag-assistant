# 医疗问答系统（RAG）

企业实训的小组项目。在课程提供的问答示例工程上改造，做成一个面向医疗场景的检索增强问答系统。

> 说明：基础框架来自课程示例工程，药品查询、症状分诊两个工具由组员实现；本人负责环境搭建、知识库构建与全部文档。

## 功能

- **知识库问答**：医疗资料切分后向量化入库，提问时先检索再交给大模型生成，回答可带出处
- **提示词约束**：只回答医疗问题、不给诊断结论、回答要有依据、超出范围直接拒绝
- **工具调用**：药品信息查询（对乙酰氨基酚、布洛芬）、按症状推荐就诊科室
- **流式输出**：前端边生成边显示

## 技术栈

Python · FastAPI · Chroma · LangChain · 阿里云百炼（通义千问 qwen-plus / text-embedding-v1）· SSE

## 目录

```
app/
├─ server.py        FastAPI 服务，/chat 走 SSE 流式
├─ rag_chat.py      文档加载 → 切分（500 字 / 重叠 100）→ 向量化 → 检索（top-3）
├─ chat.py          系统提示词 + 对话历史 + 调用通义千问
├─ tools.py         药品查询、症状分诊、时间天气等工具
├─ static/chat.html 前端对话页
└─ chroma_db/       已构建的向量库（可删除后重建）
docs/
└─ 医疗问诊知识库.pdf
requirements.txt
```

## 运行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # 填入自己的百炼 API Key
cd app
python server.py
```

浏览器打开 <http://localhost:8001/chat>。

修改知识库：把新文档放进 `docs/`，删掉 `app/chroma_db/` 再启动，会自动重新切分入库。

## 分工（六人小组）

| 内容 | 负责人 |
|---|---|
| 环境搭建、模型接口接入、知识库切分入库与检索验证 | 本人 |
| 需求 / 设计 / 部署 / 使用说明 / 答辩材料 | 本人 |
| 药品查询、症状分诊工具 | 组员 |
| 前端页面、服务端框架 | 基于课程示例工程 |

## 已知限制

- 知识库规模较小（2 页资料、4 个片段），主要验证链路可行性
- 药品库只内置了两种常见药，科室映射基于关键词规则
- `.env` 中不含真实密钥
