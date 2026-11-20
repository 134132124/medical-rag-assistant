"""
RAG知识问答 - 通过检索文档去回复对应问题
支持的文档 docx/pdf/txt
"""
import os

import dashscope
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chat import add_chat_history,CHAT_HISTORY
from tools import decide_tool,execute_tool


#加载.env中的环境配置
load_dotenv()

#获取环境配置给属性赋值
dashscope.api_key = os.getenv("API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")     #对话模型名称
EM_MODEL = os.getenv("EM_MODEL")         #向量模型名称

#文档存放位置
DOCS_DIR = "../docs"

#向量数据存放位置
VECTOR_DB_PATH = "./chroma_db"

#支持哪些文档转为向量数据
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader
}

class RAGAssistant:
    """构建知识库的各种方法以及内容相似度检索的方法"""
    def __init__(self):
        self.vectorstore = None    #用于后续保存向量数据对象，对用户问题进行答案检索

    def get_all_docs(self):
        """加载指定目录下所有支持的文件到内存中"""

        #接收所有文档的文档片段的列表
        all_documents = []

        #检查存放文件的目录是否存在
        if not os.path.exists(DOCS_DIR):
            print("存放文件的目录不存在，请检查目录是否正确")
            return

        #遍历指定目录的所有文档
        for filename in os.listdir(DOCS_DIR):
            ext = os.path.splitext(filename)[1]   #文档后缀名
            file_path = os.path.join(DOCS_DIR,filename)  #文档全路径，用于加载时使用

            if ext in LOADER_MAPPING:
                print(f"加载文档:{filename}")

                #获取加载器类型
                loader_class = LOADER_MAPPING[ext]

                #txt文件要指定加载的编码格式
                if ext ==".txt":
                    loader = loader_class(file_path,encoding="utf-8")
                else:
                    loader = loader_class(file_path)

                #加载返回文档片段列表
                documents = loader.load()
                print(f"打印文档列表的第一个元素:{documents[0]}")

                #将加载的文档列表放到总列表中
                all_documents.extend(documents)

                print(f"文档加载成功{filename}共{len(documents)}页/段")

        return all_documents

    def build_knowledge_base(self):
        """构建向量知识库  文档切割 -> 向量化 ->  持久化到向量数据库"""

        #获取所有文档片段
        all_docs = self.get_all_docs()

        if not all_docs:
            print("未找到任何支持的文档片段")
            return

        print(f"共加载{len(all_docs)}个文档片段")

        print("正在进行文本分块.........")

        text_sp = RecursiveCharacterTextSplitter(
            separators= ["\n\n", "\n", "。", "；"],   #优先将文档按照哪些符号进行分割
            chunk_size=500,                          #分割之后每段的最大字符数，超出此字符数需要再次分割
            chunk_overlap=100                        #上下块重叠的字符数量
        )

        chunks = text_sp.split_documents(all_docs)
        print(f"共分为{len(chunks)}个文档块")

        print("正在进行向量化")
        em = DashScopeEmbeddings(
            model=EM_MODEL,            #向量模型名称
            dashscope_api_key=dashscope.api_key   #调用凭证
        )

        print("正在存入向量数据库.....")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,    #文档块
            embedding = em,      #向量模型对象名称
            persist_directory=VECTOR_DB_PATH  #向量数据存放位置
        )

        #向量数据持久化存储到文件中
        self.vectorstore.persist()

        print("知识库构建完成")

    def search_answer(self,question,k=3):
        """
        根据用户问题从知识库检索相关片段
        :param question: 用户问题
        :param k: 最多检索几个相似片段
        :return: 检索到的相似片段
        """
        docs = self.vectorstore.similarity_search(question,k=k)

        return docs

    def load_knowledge_base(self):
        """
        加载向量知识库(只要文档不发生变动，只需要构建一次，后续进行知识库加载即可)
        """
        print("正在加载已有知识库")

        #向量模型对象
        em = DashScopeEmbeddings(
            model=EM_MODEL,
            dashscope_api_key= dashscope.api_key
        )

        #从知识库路径中读取向量数据
        self.vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,   #从什么地方读取
            embedding_function= em             #向量检索函数需要的模型对象
        )

        print("知识库加载成功.......")

    def init_knowledge(self):
        """初始化知识库 判断是构建知识库还是加载知识库"""
        #如果知识库目录不存在或者目录存在但是没有任何内容
        if not os.path.exists(VECTOR_DB_PATH) or not os.listdir(VECTOR_DB_PATH):
            self.build_knowledge_base()
        else:
            self.load_knowledge_base()

    def chat(self,question):
        """基于知识库检索回答用户问题"""

        #通过问题从知识库中检索答案
        docs = self.search_answer(question)

        #将从知识库中检索到的答案和用户的原始问题一块交给大模型(相当于你在给大模型提问的时候提供给大模型一个小抄)
        context_part = [] #定义列表存放答案和需要以及答案的来源

        for i,doc in enumerate(docs,1):
            source = doc.metadata.get("source","未知来源")
            context_part.append(f"【文档片段{i}-来自{source}】\n{doc.page_content}")

        #将参考列表组合成字符串
        context = "\n\n".join(context_part)

        #整合小抄和原始问题
        user_msg = f"""
            【知识库文档】
            {context}
            
            【用户问题】
            {question}
            请根据知识库的内容回答问题，并尽量标注信息来源文件
        """
        # 将用户问题放到对话列表
        add_chat_history("user", user_msg)

        resps = dashscope.Generation.call(
            model=CHAT_MODEL,  # 模型的名称
            messages=CHAT_HISTORY,  # 对话问题
            result_format="message",  # 返回值类型
            stream=True,  # 开启流式输出
            incremental_output=True  # 增量输出
        )
        # 拼接AI的回答片段
        full_answer = ""
        for resp in resps:
            if resp.status_code == 200:
                result = resp.output.choices[0].message.content  # 模型输出的答案
                print(result, end="", flush=True)  # end = "": 每个回答片段不拼接任何内容， flush=True:强制刷新缓冲区
                full_answer += result  # 拼接所有回答片段
            else:
                print(f"模型调用失败:{resp}")
        # 将AI的回复内容放到对话历史
        add_chat_history("assistant", full_answer)

    def chat_stream(self, question):
        """基于知识库检索和工具调用回答用户问题"""

        tool_name,params = decide_tool(question)
        #如果需要调用工具，就不需要检索知识库，否则相反
        if tool_name:
            print(f"工具调用{tool_name},参数:{params}")
            tool_result = execute_tool(tool_name,**params)
            user_msg = f"""
                【工具调用结果】
                {tool_result}
                【用户问题】
                {question}
                请根据工具调用结果回答用户问题
            """
        else:
            # 通过问题从知识库中检索答案
            docs = self.search_answer(question)

            # 将从知识库中检索到的答案和用户的原始问题一块交给大模型(相当于你在给大模型提问的时候提供给大模型一个小抄)
            context_part = []  # 定义列表存放答案和需要以及答案的来源

            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                context_part.append(f"【文档片段{i}-来自{source}】\n{doc.page_content}")

            # 将参考列表组合成字符串
            context = "\n\n".join(context_part)

            # 整合小抄和原始问题
            user_msg = f"""
                 【知识库文档】
                 {context}
    
                 【用户问题】
                 {question}
                 请根据知识库的内容回答问题，并尽量标注信息来源文件
             """

        # 将用户问题放到对话列表
        add_chat_history("user", user_msg)

        resps = dashscope.Generation.call(
            model=CHAT_MODEL,  # 模型的名称
            messages=CHAT_HISTORY,  # 对话问题
            result_format="message",  # 返回值类型
            stream=True,  # 开启流式输出
            incremental_output=True  # 增量输出
        )
        # 拼接AI的回答片段
        full_answer = ""
        for resp in resps:
            if resp.status_code == 200:
                result = resp.output.choices[0].message.content  # 模型输出的答案
                # print(result, end="", flush=True)  # end = "": 每个回答片段不拼接任何内容， flush=True:强制刷新缓冲区
                yield result     #yield:返回多次内容
                full_answer += result  # 拼接所有回答片段
            else:
                # print(f"模型调用失败:{resp}")
                return f"模型调用失败:{resp}"
        # 将AI的回复内容放到对话历史
        add_chat_history("assistant", full_answer)




#定义变量接收Rag对象
_rag = None

def get_rag():
    """获取rag对象并返回该对象的单例模式"""
    #声明_rag是全局变量
    global _rag

    if _rag is None:
        _rag = RAGAssistant()
        _rag.init_knowledge()  #初始化知识库

    return _rag




def main():
    """主函数"""
    #检查API_KEY是否配置
    if not dashscope.api_key:
        print("错误: 未找到API Key")
        return

    print("="*30)
    print("智能医疗健康系统")
    print("="*30)
    #创建Rag对象
    rag = RAGAssistant()
    #初始化知识库
    rag.init_knowledge()

    while True:
        #获取用户输入的信息
        user_input = input("\n 用户: ")

        if "/exit" == user_input:
            break

        #调用对话方法
        print("\n AI助手: ")
        rag.chat(user_input)

        print()

if __name__ == "__main__":
    main()