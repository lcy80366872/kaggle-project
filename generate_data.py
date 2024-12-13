from openai import OpenAI
import os, math, numpy as np
import json
import pandas as pd
import re
PROMPT  = """
You are a mathematician. You need to generate data in the following format:
- Question: a math problem question.
- Answers A: right choice answers.
- Answers B: wrong choice answers.(no need to explain it.)
- Given misconception: {Misconception}
- Subject: the category of the math question problem.
- ConstructName: Most granular level of knowledge related to question.

Here is an example:
######
Question: Question you need to generate.
Answers A: Right answer you need to generate.
Answers B: Wrong answer you need to generate.
Given Misconception: A misconception you are given.
Subject: the category of the math question problem you generate.
ConstructName: Most granular level of knowledge related to question. 
######
According to the Given misconception, your task is to  create a new `Question` ,`Answers A` , `Answers B` ,`Subject` and `ConstructName`.
The logic behind the wrong answer(Answers B) should be the given misconception.Your output should strictly follow example, and nothing else should be output at the same time.
"""
PROMPT1  = """
You are an expert mathematician. You need to generate data in the following format:
- Question: a math problem question.
- Answers A: right choice answers.
- Answers B: wrong choice answers.(no need to explain it.)
- Given misconception: {Misconception}
- Subject: the category of the math question problem.
- ConstructName: Most granular level of knowledge related to question.

Here is an example:
######
Question: Question you need to generate.
Answers A: Right answer you need to generate.
Answers B: Wrong answer you need to generate.
Given Misconception: A misconception you are given.
Subject: the category of the math question problem you generate.
ConstructName: Most granular level of knowledge related to question. 
######
According to the Given misconception, your task is to  create a new `Question` ,`Answers A` , `Answers B` ,`Subject` and `ConstructName`.
The logic behind the wrong answer(Answers B) should be the given misconception.Your output should strictly follow example, and nothing else should be output at the same time.
"""
PROMPT2  = """
You are an expert mathematician. You need to generate data in the following format:
- Question: a math problem question.
- Answers A: right choice answers.
- Answers B: wrong choice answers.(no need to explain it.)
- Given misconception: {Misconception}
- Subject: the category of the math question problem.
- ConstructName: Most granular level of knowledge related to question.

According to the Given misconception, your task is to  create a new `Question` ,`Answers A` , `Answers B` ,`Subject` and `ConstructName`.
The logic behind the wrong answer(Answers B) should be the given misconception.Your output should strictly follow example, and nothing else should be output at the same time.
"""
def get_response(prompt):

    client = OpenAI(
        api_key="sk-a59f42281e31415eaf38c8331475e911", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen2-72b-instruct",#qwen2-72b-instruct",  llama3.1-70b-instruct
        temperature=0,
        messages=[
            {'role': 'system', 'content': 'You are a mathematical expert'},
            {'role': 'user', 'content': prompt}
        ])
    result = json.loads( completion.model_dump_json())
    content = result['choices'][0]['message']['content']
    print(content)
    # print("#########")

    return content

def apply_template(row,prompt):
    text=prompt.format(Misconception=row['MisconceptionName'])
    return text


def extract_data(text):
    # 提取 ConstructName
    construct_name = re.search(r'ConstructName:\s*(.*?)(?=\n|$)', text)
    construct_name = construct_name.group(1) if construct_name else None

    # 提取 SubjectName
    subject_name = re.search(r'Subject:\s*(.*?)(?=\n|$)', text)
    subject_name = subject_name.group(1) if subject_name else None

    answer_a_text = re.search(r'Answers A:\s*(.*?)(?=\n|$)', text)
    answer_a_text = answer_a_text.group(1) if answer_a_text else None

    answer_b_text = re.search(r'Answers B:\s*(.*?)(?=\n|$)', text)
    answer_b_text = answer_b_text.group(1) if answer_b_text else None

    # 提取QuestionText
    question_text = re.search(r'Question:\s*(.*?)(?=\n|$)', text)
    question_text = question_text.group(1) if question_text else None

    # 提取 MisconceptionB
    misconception_b = re.search(r'Given Misconception:\s*(.*?)(?=\n|$)', text)
    misconception_b = misconception_b.group(1) if misconception_b else None
    correct_answer = 'A'

    return construct_name, subject_name, correct_answer, answer_b_text, answer_a_text, question_text, misconception_b

def main(name,prompt):
    df=pd.read_csv("misconception_mapping_non2.csv")
    df["Prompt"] = df.apply(lambda row: apply_template(row,prompt), axis=1)
    responses=[]
    print("infering")
    n=0
    for i in df["Prompt"]:
        print(n)
        responses.append(get_response(i))
        n=n+1
    # responses = [x for x in responses]
    df["FullResponse"] = responses
    data_b = []

    # 对表格A中的每一行应用提取函数，并保留原始内容
    for _, row in df.iterrows():
        text = row['FullResponse']
        MisconceptionId = row['MisconceptionId']
        Misconception = row['MisconceptionName']
        construct_name, subject_name, correct_answer, answer_b_text, answer_a_text, question_text, misconception_b = extract_data(
            text)
        data_b.append({
            'MisconceptionId': MisconceptionId,
            'MisconceptionName': Misconception,
            'ConstructName': construct_name,
            'SubjectName': subject_name,
            'AnswerAText': answer_a_text,
            'AnswerBText': answer_b_text,
            'QuestionText': question_text,
            'MisconceptionB': misconception_b
        })

    # 将数据保存为表格B.csv
    df_b = pd.DataFrame(data_b)
    df_b.to_csv(name, index=False, encoding='utf-8')
    print("extern_data.csv生成完毕！")

if __name__ == '__main__':
    main('extern_data_more1.csv',PROMPT)
    main('extern_data_more2.csv',PROMPT2)