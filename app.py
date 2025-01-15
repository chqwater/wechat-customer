import os
import time

import xmltodict
from flask import Flask, request

from bot import Chatbot

app = Flask(__name__)

# 初始化 chatbot，不需要传入 api_key，让它从环境变量获取
try:
    chatbot = Chatbot()
    # 验证是否成功初始化
    if chatbot.api_key:
        print("Chatbot successfully initialized with API key")
    else:
        print("Warning: Chatbot initialized without API key")
except ValueError as e:
    print(f"Error initializing chatbot: {e}")
    # 这里可以添加适当的错误处理

def parse_msg(xml: bytes) -> dict:
    return xmltodict.parse(xml).get('xml')

def pack_msg(reply_info: dict, msg: str) -> str:
    resp_dict = {
        "xml": {
            'ToUserName': reply_info.get('FromUserName'),
            'FromUserName': reply_info.get('ToUserName'),
            'CreateTime': int(time.time()),
            'MsgType': 'text',
            'Content': msg
        }
    }
    return xmltodict.unparse(resp_dict)

@app.route("/wechat", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # For verifying the server in official account platform
        return request.args.get('echostr')

    def answer(ask: str) -> str:
        response = chatbot.ask(ask, temperature=0.5)
        print("Ask: " + ask)
        response_text = response["choices"][0]["message"]["content"]
        print("ChatGPT: " + response_text)
        return response_text

    if not request.data:
        return '', 403

    reply_info = parse_msg(request.data)

    user_content = reply_info['Content']
    #msg = f"你刚刚发送了【{user_content}】"

    msg = answer(user_content)

    return pack_msg(reply_info, msg)


def confirmation_session(content: str) -> bool:
    return None

if __name__ == '__main__':
    # 启动前验证环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=80, debug=True)
