import csv
import requests
from sklearn.metrics import jaccard_score
import jieba
import textstat
import numpy as np
import logging
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 小胰宝RAG服务的API端点和密钥
api_endpoint = "http://ai.xiaofeibao.com.cn/v1/chat-messages"
api_key = "app-##"
user_id = "abc-123"  # 用户ID
conversation_id = ""  # 会话ID

# 读取QA文件
qa_file = "/Users/qinxiaoqiang/Downloads/测试问题.csv"
output_file = "/Users/qinxiaoqiang/Downloads/生成答案记录.csv"

def read_qa_file(file_path):
    questions = []
    answers = []
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        # 打印列名，确认是否正确
        logging.info(f"列名: {reader.fieldnames}")
        for row in reader:
            question = row.get('Question', '').strip()
            answer = row.get('answer', '').strip()
            if question:  # 确保问题不为空
                questions.append(question)
                answers.append(answer)
    return questions, answers

def generate_answer(question, api_endpoint, api_key, user_id, conversation_id):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": {},
        "query": question,
        "user": user_id,
        "response_mode": "blocking",
        "conversation_id": conversation_id
    }
    response = requests.post(api_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        generated_answer = response.json().get("answer", "")
        logging.info(f"Question: {question}, Generated Answer: {generated_answer}")
        return generated_answer
    else:
        logging.error(f"Error: {response.status_code} - {response.text}")
        return ""

def process_questions_in_batches(questions, batch_size, api_endpoint, api_key, user_id, conversation_id):
    generated_answers = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}/{len(questions) // batch_size + 1}")
        generated_batch = [generate_answer(q, api_endpoint, api_key, user_id, conversation_id) for q in batch]
        generated_answers.extend(generated_batch)
    return generated_answers

def write_output_file(questions, answers, generated_answers, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Original Answer', 'Generated Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(questions)):
            writer.writerow({
                'Question': questions[i],
                'Original Answer': answers[i],
                'Generated Answer': generated_answers[i]
            })

def calculate_accuracy(answers, generated_answers):
    accuracy = []
    for i in range(len(answers)):
        tokenized_answer = list(jieba.cut(answers[i].lower()))
        tokenized_generated_answer = list(jieba.cut(generated_answers[i].lower()))
        if tokenized_answer and tokenized_generated_answer:
            accuracy.append(jaccard_score(tokenized_answer, tokenized_generated_answer, average='macro'))
        else:
            accuracy.append(0.0)
    return accuracy

def calculate_relevance(questions, generated_answers, api_endpoint, api_key, user_id, conversation_id):
    relevance = []
    for i in range(len(questions)):
        data = {
            "inputs": {},
            "query": questions[i],
            "user": user_id,
            "response_mode": "blocking",
            "conversation_id": conversation_id
        }
        response = requests.post(api_endpoint, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }, json=data)
        if response.status_code == 200:
            response_data = response.json()
            question_embedding = np.array(response_data.get("question_embedding", []))
            answer_embedding = np.array(response_data.get("answer_embedding", []))
            if question_embedding.size > 0 and answer_embedding.size > 0:
                relevance.append((question_embedding @ answer_embedding.T) / (np.linalg.norm(question_embedding) * np.linalg.norm(answer_embedding)))
            else:
                relevance.append(0.0)
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            relevance.append(0.0)
    return relevance

def calculate_readability(generated_answers):
    readability = []
    for generated_answer in generated_answers:
        readability.append(textstat.flesch_kincaid_grade(generated_answer))
    return readability

def test_api_call():
    test_question = "你好，介绍下小细胞肺癌的诊断概要方法？"
    generated_answer = generate_answer(test_question, api_endpoint, api_key, user_id, conversation_id)
    logging.info(f"Test Question: {test_question}, Test Generated Answer: {generated_answer}")

def main():
    # 运行测试环节
    test_api_call()

    questions, answers = read_qa_file(qa_file)
    generated_answers = process_questions_in_batches(questions, batch_size=10, api_endpoint=api_endpoint, api_key=api_key, user_id=user_id, conversation_id=conversation_id)
    write_output_file(questions, answers, generated_answers, output_file)
    accuracy = calculate_accuracy(answers, generated_answers)
    relevance = calculate_relevance(questions, generated_answers, api_endpoint, api_key, user_id, conversation_id)
    readability = calculate_readability(generated_answers)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Relevance: %s", relevance)
    logging.info("Readability: %s", readability)

if __name__ == "__main__":
    main()
