"""
Usage: 
python tests/test_framework.py
"""

from openai import OpenAI
import json
import os


test_cases = None
with open("./tests/ground_truth.json", "r") as file:
    test_cases = json.load(file)

mock_chatbot_answers = None
with open("./tests/mock_chatbot_answers.json", "r") as file:
    mock_chatbot_answers = json.load(file)


# Function to filter JSON by question
def filter_by_question(data, question):
    return [item["answer"] for item in data if item["question"] == question][0]


def ask_chatbot(question):
    return filter_by_question(mock_chatbot_answers, question)


# Initialize OpenAI client with your API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

test_results = []
for test_case in test_cases:
    question = test_case["question"]
    expected_answer = test_case["answer"]

    chatbot_answer = ask_chatbot(question)

    test_instructions = f"""
    You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Expert]: {expected_answer}
    ************
    [Submission]: {chatbot_answer}
    ************
    [END DATA]

    Compare the factual content of the submitted answer with the expert answer.

    Ignore any differences in style, grammar, or punctuation.

    The submitted answer may either be a subset or superset of the expert answer,
    or it may conflict with it. Determine which case applies.

    Answer the question by selecting one of the following options and only output A, or B, or C, or D, or E:
    (A) The submitted answer contains all the same details as the expert answer.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (D) The answers differ, but these differences don't matter from the perspective of factuality.
    (E) There is a disagreement between the submitted answer and the expert answer.
    """

    test_result = client.chat.completions.create(
        messages=[{"role": "user", "content": test_instructions}],
        model="gpt-4o-2024-05-13",
    )
    test_result = test_result.choices[0].message.content

    test_results.append(
        {
            "question": question,
            "expected_answer": expected_answer,
            "chatbot_answer": chatbot_answer,
            "test_result": test_result,
        }
    )
    print(
        f"question: {question}\nexpected_answer: {expected_answer}\nchatbot_answer: {chatbot_answer}\ntest_result: {test_result}\n"
    )

# Write the array of objects to a JSON file
with open("./tests/test_results.json", "w") as file:
    json.dump(test_results, file, indent=4)
