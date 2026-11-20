"""
工具模块 - 用于封装接入到大模型的各个工具
"""
import json
import os
import sys
from datetime import datetime

import dashscope
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import Dict, Any


@tool()
def query_drug_info(drug_name: str) -> Dict[str, Any]:
    """
    查询药品信息
    :param drug_name: 药品通用名，字符串
    :return: 结构化药品信息字典
    """
    drug_db = {
        "对乙酰氨基酚": {
            "通用名": "对乙酰氨基酚",
            "适应症": "用于普通感冒或流行性感冒引起的发热，也用于缓解轻至中度疼痛如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。",
            "用法用量": "口服。成人一次1片，若持续发热或疼痛，可间隔4-6小时重复用药一次，24小时内不得超过4次。",
            "禁忌": "严重肝肾功能不全者禁用；对本品过敏者禁用。",
            "不良反应": "偶见皮疹、荨麻疹、药热及粒细胞减少。长期大量用药会导致肝肾功能异常。",
            "注意事项": "本品为对症治疗药，用于解热连续使用不超过3天，用于止痛不超过5天。"
        },
        "布洛芬": {
            "通用名": "布洛芬",
            "适应症": "用于缓解轻至中度疼痛如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。也用于普通感冒或流行性感冒引起的发热。",
            "用法用量": "口服，成人一次1粒，一日2次（早晚各一次）。",
            "禁忌": "对其他非甾体抗炎药过敏者禁用；孕妇及哺乳期妇女禁用；对阿司匹林过敏的哮喘患者禁用。",
            "不良反应": "少数病人可出现恶心、呕吐、胃烧灼感或轻度消化不良、胃肠道溃疡及出血、头痛、头晕、耳鸣。",
            "注意事项": "本品为对症治疗药，不宜长期或大量使用。"
        }
    }

    if drug_name not in drug_db:
        return {
            "code": 404,
            "msg": f"未查询到【{drug_name}】相关药品信息，请确认药品通用名是否正确",
            "data": None
        }
    return {
        "code": 200,
        "msg": "success",
        "data": drug_db[drug_name]
    }


@tool()
def query_department(symptom: str) -> Dict[str, Any]:
    """
    根据症状推荐就诊科室
    :param symptom: 用户症状描述文本
    :return: 科室推荐结构化结果
    """
    symptom = symptom.lower()
    # 症状-科室映射规则
    rule_map = [
        {"keywords": ["发烧", "咳嗽", "咽痛", "胸闷", "咳痰", "感冒"], "department": "呼吸内科"},
        {"keywords": ["胃痛", "反酸", "腹胀", "腹泻", "便血"], "department": "消化内科"},
        {"keywords": ["头痛", "头晕", "肢体麻木", "抽搐"], "department": "神经内科"},
        {"keywords": ["心慌", "心悸", "胸痛", "高血压"], "department": "心血管内科"},
        {"keywords": ["骨折", "腰腿痛", "关节痛", "扭伤"], "department": "骨科"},
        {"keywords": ["皮疹", "瘙痒", "痤疮", "湿疹"], "department": "皮肤科"},
        {"keywords": ["眼干", "眼痛", "视力下降", "红眼"], "department": "眼科"},
        {"keywords": ["耳朵疼", "耳鸣", "听力下降", "鼻塞"], "department": "耳鼻喉科"},
    ]

    recommend_list = []
    for rule in rule_map:
        for kw in rule["keywords"]:
            if kw in symptom:
                recommend_list.append(rule["department"])
                break

    # 去重
    recommend_list = list(set(recommend_list))
    if not recommend_list:
        return {
            "code": 404,
            "msg": "无法根据当前症状匹配科室，建议补充详细症状，或前往全科/普通内科初步就诊",
            "data": {"recommend_department": []}
        }

    return {
        "code": 200,
        "msg": "success",
        "data": {"recommend_department": recommend_list}
    }


@tool
def get_current_datetime() -> str:
    """
    获取当前的日期时间
    当用户询问 "今天几号"、"现在几点"、"当前时间"、"今天周几"等问题的时候使用
    :return: 格式化之后的日期时间字符串
    """

    now = datetime.now()

    return now.strftime('%Y-%m-%d %H:%M:%S')


@tool
def get_weather(city: str) -> str:
    """
    查询某城市的实时天气，当用户询问某个城市天气状况时使用
    :param city: 中文城市名称 如北京、上海、威海
    :return: 指定城市的天气情况
    """
    try:
        import httpx

        resp = httpx.get(f"https://wttr.in/{city}", params={"format": "%C %t", "lang": "zh"}, timeout=5)
        return f"{city}当前天气:{resp.text}"
    except Exception:
        return f"暂时查不到{city}的天气(网络可能受限)"


# 工具列表
TOOLS = [
    get_weather,
    get_current_datetime,
    query_drug_info,
    query_department
]

# 加载环境配置
load_dotenv()

dashscope.api_key = os.getenv("API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")


def decide_tool(question: str) -> tuple:
    """
    创建智能体，根据用户问题来决定是否调用工具，以及从问题中获取工具执行参数
    :param question: 用户问题
    :return: (工具名称,工具执行参数) 或 (None,None)
    """
    tool_list = []  # 工具明细获取

    for tool in TOOLS:
        tool_list.append({
            "name": tool.name,
              "description": tool.description,
            "parameters": tool.args
        })

    prompt = f"""
        你是一个智能助手，根据用户问题来决定是否调用工具。
        可用工具:
        {json.dumps(tool_list, ensure_ascii=False, indent=2)} 
        
        用户问题:{question}
        请严格以纯JSON的形式输出决策结果，不要包含任何描述性文字
        {{ "tool":"工具名称","params":{{"参数名":"参数值"}} }}
        
        如果不需要调用任何工具，输出:
        {{"tool":"none","params":{{}} }}
    """
    resp = dashscope.Generation.call(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}],
        result_format="messages"
    )
    if resp.status_code == 200:
        content = resp.output.choices[0].message.content
        print(f"工具调用的原始返回:{content}")

        result = json.loads(content)
        tool_name = result.get("tool", "none")
        params = result.get("params", {})

        if tool_name != "none":
            return tool_name, params
    else:
        print(f"工具使用决策失败:{resp}")

    return None, None


def execute_tool(tool_name: str, **kwargs) -> str:
    """
    根据工具名称执行对应的工具函数获得结果
    :param tool_name:工具名称
    :param kwargs:工具执行需要的可变参数
    :return:工具执行结果
    """
    for tool in TOOLS:
        if tool.name == tool_name:
            return tool.invoke(kwargs)
    return f"未找到可执行的工具{tool_name}"


if __name__ == "__main__":

    tool_name, params = decide_tool("布洛芬")
    print(tool_name, params)
    print(execute_tool(tool_name, **params))


    # tool_name, params = decide_tool("威海天气怎么样")
    # print(tool_name, params)

