# 调用API和模型进行对话，支持对话记忆和流式输出

import os
from platform import system

import dashscope
from dotenv import load_dotenv

# 加载.env的环境配置↓
load_dotenv()

dashscope.api_key = os.getenv("API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# 系统提示词，设置模型的角色，模型具备的功能和注意事项↓
SYSTEM_MESSAGE = """
                你是专业的医疗健康助手，具备丰富的医学知识
                规则：
                1.只回答医疗健康相关问题（疾病预防、症状分析建议、用药指南、养生保健
                2.不提供诊断结论，引导用户及时就医
                3.回答专业、准确、有据可查
                4.超出医疗健康范围的问题礼貌拒绝       
"""

# 定义变量，保存对话历史↓
CHAT_HISTORY = [
    {"role": "system", "content": SYSTEM_MESSAGE}
]


def add_chat_history(role, content):
    """
    将对话放到对话历史列表
    :param role:角色(system, user,assistant)
    :param content:对话内容
    :return:
    """
    CHAT_HISTORY.append(
        {"role": role, "content": content}
    )


def chat(prompt):
    """
    对话函数
    :param prompt:用户提问的问题
    :return:
    """
    # 和模型对话的列表↓
    msg = [
        {"role": "system", "content": SYSTEM_MESSAGE},  # 系统提示词
        {"role": "user", "content": prompt}  # 用户问题
    ]
    # 将用户问题放到对话列表↓
    add_chat_history("user", prompt)

    resps = dashscope.Generation.call(
        model=CHAT_MODEL,  # 模型的名称
        messages=CHAT_HISTORY,  # 对话问题
        result_format="message",  # 返回值类型
        stream=True,  # 开启流式输出
        incremental_output=True  # 增量输出
    )
    # 拼接AI的回答片段↓
    full_answer = " "

    for resp in resps:
        if resp.status_code == 200:
            result = resp.output.choices[0].message.content  # 模型输出的答案
            print(result, end=" ", flush=True)  # end=":每个回答片段不拼接任何内容，flush=True:强制刷新缓冲区
            full_answer += result  # 拼接所有回答片段
        else:
            print(f"模型调用失败:{resp}")
    # 将AI的回复内容放到对话历史↓
    add_chat_history("system", full_answer)


def main():
    """
    主函数
    :return:
    """
    # 检查API_KEY是否配置↓
    if not dashscope.api_key:
        print("错误:未找到API Key")
        return

    print("=" * 30)
    print("智能医疗健康系统")
    print("=" * 30)

    while True:
        # 获取用户输入的信息↓
        user_input = input("\n 用户: ")

        if "/exit" == user_input:
            break

        # 调用对话方法↓
        print("\n AI助手: ")
        chat(user_input)
        print()


if __name__ == "__main__":
    main()
