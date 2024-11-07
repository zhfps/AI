import json
import ollama
import asyncio

# 模拟航班数据库
def get_flight_info(departure: str, arrival: str) -> str:
    flights = {
        "北京-上海": {"起飞": "08:00", "到达": "10:30", "航程": "2小时30分"},
        "上海-北京": {"起飞": "14:00", "到达": "16:30", "航程": "2小时30分"},
        "广州-成都": {"起飞": "10:00", "到达": "12:30", "航程": "2小时30分"},
        "成都-广州": {"起飞": "15:00", "到达": "17:30", "航程": "2小时30分"},
        "深圳-杭州": {"起飞": "09:00", "到达": "11:00", "航程": "2小时"},
        "杭州-深圳": {"起飞": "13:00", "到达": "15:00", "航程": "2小时"}
    }

    key = f'{departure}-{arrival}'
    return json.dumps(flights.get(key, {'错误': '未找到航班信息'}), ensure_ascii=False)

async def run(model: str):
    client = ollama.AsyncClient()
    # 初始化对话
    messages = [{'role': 'user', 'content': '广州到成都的航班时间是什么时候？'}]

    # 第一次调用：发送查询和函数描述
    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_flight_info',
                    'description': '获取两个城市之间的航班信息',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'departure': {
                                'type': 'string',
                                'description': '出发城市',
                            },
                            'arrival': {
                                'type': 'string',
                                'description': '到达城市',
                            },
                        },
                        'required': ['departure', 'arrival'],
                    },
                },
            },
        ],
    )

    # 将模型响应添加到对话历史
    messages.append(response['message'])

    # 检查模型是否选择使用函数
    if not response['message'].get('tool_calls'):
        print("模型没有使用函数，直接回复：")
        print(response['message']['content'])
        return

    # 处理函数调用
    if response['message'].get('tool_calls'):
        available_functions = {
            'get_flight_info': get_flight_info,
        }
        for tool in response['message']['tool_calls']:
            function_name = tool['function']['name']
            function_args = tool['function']['arguments']

            # 处理参数可能是字典或字符串的情况
            if isinstance(function_args, dict):
                args_dict = function_args
            else:
                try:
                    args_dict = json.loads(function_args)
                except json.JSONDecodeError:
                    print("无法解析函数参数")
                    return

            try:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(
                    args_dict['departure'],
                    args_dict['arrival']
                )
                messages.append(
                    {
                        'role': 'tool',
                        'content': function_response,
                    }
                )
            except KeyError as e:
                print(f"缺少必要参数: {e}")
                return
            except Exception as e:
                print(f"处理过程中出现错误: {e}")
                return

    # 第二次调用：获取最终响应
    final_response = await client.chat(model=model, messages=messages)
    print(final_response['message']['content'])

# 运行程序
if __name__ == "__main__":
    asyncio.run(run('qwen2.5:72b'))