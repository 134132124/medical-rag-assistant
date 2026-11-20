"""
智能对话助手 API服务
"""
import json

import uvicorn
from fastapi import FastAPI,Request,Response
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from rag_chat import get_rag

#静态文件存在的目录
STATIC_DIR = "./static"

#创建FastAPI应用对象
app = FastAPI(title="智能对话助手",description="基于知识库问答和工具调用的智能人事问答助手")

#挂载静态文件目录
app.mount("/static",StaticFiles(directory=STATIC_DIR),name="static")



@app.get("/")
async def root():
    return RedirectResponse(url="/static/chat.html")

@app.post("/chat")
async def chat_stream(request:Request):
    """
    流式对话接口，接收用户问题并返回流式数据
    :param request: 请求对象
    :return: 大模型的回答片段
    """
    #get请求获取参数
    # question = request.query_params.get("question")
    #post请求获取参数
    body = await request.json()
    question = body.get("question")


    if not question:
        return {"error":"问题不能为空"}

    #创建rag对象
    rag = get_rag()

    def generation():
        """获取模型的返回值并处理为流式数据格式"""
        chunk_count = 0   #记录当前回答由几个回答片段组成
        for chunk in rag.chat_stream(question):
            chunk_count+=1
            #将回答片段组装成json格式的sse片段输出 content:回答片段 done: 回答结束标识
            data = json.dumps({"content":chunk,"done":False},ensure_ascii=False)
            yield f"data: {data}\n\n"   #流式输出的格式 data:后面要跟上空格，以\n\n结尾
        print(f"流式输出完成。共输出{chunk_count}个回答片段")

        done_data = json.dumps({"content":"","done":True},ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        generation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  #禁用缓存，让客户端直接展示在页面上
            "Connection": "keep-alive"   #持续保持连接
        }
    )



if __name__ == "__main__":
    print("="*50)
    print("启动认识小助手API服务")
    print("="*50)
    print("测试地址，直接点击访问: http://localhost:8001/")
    # print("接口测试路径: http://localhost:8001/chat?question=你的问题")

    uvicorn.run(app,host="0.0.0.0",port=8001)