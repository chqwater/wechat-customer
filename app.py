import os
import time

import xmltodict
from flask import Flask, request
import requests

from bot import Chatbot

app = Flask(__name__)

try:
    chatbot = Chatbot()

    if chatbot.api_key:
        print("Chatbot successfully initialized with API key")
    else:
        print("Warning: Chatbot initialized without API key")
except ValueError as e:
    print(f"Error initializing chatbot: {e}")

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
    user_id = parse_msg(request.data).get('FromUserName')
    def answer(ask: str) -> str:
        print(f"Current API key: {chatbot.api_key[:6]}...{chatbot.api_key[-4:]}")
        print(f"Environment API key: {os.environ.get('OPENAI_API_KEY')[:6]}...{os.environ.get('OPENAI_API_KEY')[-4:]}")
        
        response = chatbot.ask(ask, temperature=0.5)
        print("Ask: " + ask)
        response_text = response["choices"][0]["message"]["content"]

        return response_text

    if not request.data:
        return '', 403

    reply_info = parse_msg(request.data)

    user_content = f'{user_id}:' + reply_info['Content']
    msg = f"【{user_content}】"

    #msg = answer(user_content)

    return pack_msg(reply_info, msg)


def confirmation_session(content: str) -> bool:
    return None

def get_access_token(appid: str, secret: str) -> str:
    global access_token_cache
    current_time = time.time()

    if access_token_cache["token"] and access_token_cache["expire_at"] > current_time:
        return access_token_cache["token"]

    url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={appid}&secret={secret}"
    response = requests.get(url)
    data = response.json()

    if "access_token" in data:
        access_token_cache["token"] = data["access_token"]
        access_token_cache["expires_at"] = current_time + data["expires_in"] - 60
        return access_token_cache["token"]
    else:
        raise ValueError(f"Failed to get access_token: {data}")


def send_single_msg(to_user: str, content: str):
    access_token = get_access_token()
    url = f'https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={access_token}'
    payload = {
        'touser': to_user,
        'msgtype': 'text',
        'text': {
            'content': content
        }
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)


if __name__ == '__main__':
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=80, debug=True)
