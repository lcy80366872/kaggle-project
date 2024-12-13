

import vllm
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List
import torch
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import re

# model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
# tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_text(x):
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x

PROMPT  = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
Pick the correct misconception number from the below:

{Retrival}
"""
# just directly give your answers.

def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user", 
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"incorrect_answer"],
                    CorrectAnswer=row[f"correct_answer"],
                    Retrival=row[f"retrieval"]
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


misconception_df = pd.read_csv("input/misconception_mapping.csv")

df = pd.read_parquet("file/df.parquet")
indices = np.load("file/indices.npy")
#indices是个二维数组，行对应第几个数据，列对应召回的25个id
# model_path = "Qwen/Qwen2.5-32B-Instruct-AWQ"
model_path = "Qwen/Qwen2.5-72B-Instruct-AWQ"
# llm = vllm.LLM(
#     model_path,
#     quantization="awq",
#     tensor_parallel_size=2,
#     gpu_memory_utilization=0.90, 
#     trust_remote_code=True,
#     dtype="half", 
#     enforce_eager=True,
#     max_model_len=5120,
#     disable_log_stats=True
# )
llm = vllm.LLM(
    model_path,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.98, 
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=2000,
    disable_log_stats=True,
    cpu_offload_gb=8,
    swap_space=1,
    device='cuda',
    max_num_seqs=20
)
tokenizer = llm.get_tokenizer()

#根据indice，把那些编号对应的具体误解加载进来，也相当于二维数组
def get_candidates(c_indices):
    candidates = []

    mis_names = misconception_df["MisconceptionName"].values
    for ix in c_indices:
        c_names = []
        for i, name in enumerate(mis_names[ix]):
            c_names.append(f"{i+1}. {name}")  

        candidates.append("\n".join(c_names))
        
    return candidates
survivors = indices[:, -1:]

for i in range(3):
    #一次从indice里选8个顺序的误解，比如先从倒数8个挑，然后从倒数16到倒数第8挑，然后从前八个挑
    c_indices = np.concatenate([indices[:, -8*(i+1)-1:-8*i-1], survivors], axis=1)
    
    df["retrieval"] = get_candidates(c_indices)
    df["text"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)
    
    print("Example:")
    print(df["text"].values[0])
    print()
    
    responses = llm.generate(
        df["text"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777, # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=1,  # Maximum number of tokens to generate per output sequence.
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])]
        ),
        use_tqdm=True
    )
    
    responses = [x.outputs[0].text for x in responses]
    df["response"] = responses
    
    
    llm_choices = df["response"].astype(int).values - 1
    #选出8个中那个llm认为最契合的那个
    survivors = np.array([cix[best] for best, cix in zip(llm_choices, c_indices)]).reshape(-1, 1)

print("survivors ",survivors )

results = []

for i in range(indices.shape[0]):
    ix = indices[i]
    llm_choice = survivors[i, 0]
    
    results.append(" ".join([str(llm_choice)] + [str(x) for x in ix if x != llm_choice]))


df["MisconceptionId"] = results
df.to_csv("submission.csv", columns=["QuestionId_Answer", "MisconceptionId"], index=False)