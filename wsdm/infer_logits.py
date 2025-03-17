import pandas as pd
import numpy as np
import os
from datasets import Dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

run_eval = True ## To check score on train data- 100 samples



import sys
import re
import gc
import vllm
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor


system_prompt='''You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query. Determine which response is more likely to be selected by a user based on the following criteria:
- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

'''

system_prompt_chatformat='''system
You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query. Determine which response is more likely to be selected by a user based on the following criteria:
- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

'''


IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

if IS_SUBMISSION:
    df = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
else:
    ds = Dataset.from_parquet("data/full_swap_open-model.parquet")
    folds = [
    (
        [i for i in range(len(ds)) if i % 5 != fold_idx],
        [i for i in range(len(ds)) if i % 5 == fold_idx]
    )for fold_idx in range(5)]

    train_idx, eval_idx = folds[0]

    train_ds = ds.select(train_idx)
    valid_ds = ds.select(eval_idx)
    train_df = ds.to_pandas()
    valid_df = valid_ds.to_pandas()
    # n = min(32, len(train_df))
    # n = min(n, len(valid_df))
    # train_df = train_df.head(n).reset_index(drop=True)
    # valid_df = valid_df.head(n).reset_index(drop=True)
    df = train_df.copy()
    print('Length of df=',len(df))



def preprocess_function_truncation(row, tokenizer,max_length,system_prompt,system_prompt_chatformat):
    # max_length =  max_length-row['examples_tokens_len']
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    system_token= [tokenizer.bos_token_id]+tokenizer(system_prompt_chatformat,add_special_tokens=False) ["input_ids"]+[tokenizer.eos_token_id]
    user_token=tokenizer("user\n", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer('Which response is more likely to be selected by a user? (A or B)\n', add_special_tokens=False)["input_ids"]
    p=row['prompt']
    ra= row['response_a']
    rb =row['response_b']
    one_input_ids =system_token
    prev_tokens_num =  len(system_token)+len(final_p_tokens)+len(user_token)
    prompt=f'''**Here is your input to process now-**\nInput:\n\n<Query>\n{p}'''
    response_a=f'''\n<Response_A>\n{ra}'''
    response_b=f'''\n<Response_B>\n{rb}'''
    p_tokens  = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ra_tokens = tokenizer(response_a, add_special_tokens=False)["input_ids"]

    rb_tokens = tokenizer(response_b, add_special_tokens=False)["input_ids"]
    a_end_token=tokenizer(f"""\n</Response_A>\n{'---'*10}\n""", add_special_tokens=False)["input_ids"]
    b_end_token=tokenizer(f"""\n</Response_B>\n\n""", add_special_tokens=False)["input_ids"]
    p_end_token=tokenizer(f"""\n</Query>\n{'---'*10}\n""", add_special_tokens=False)["input_ids"]
    all_tokens_num = prev_tokens_num +  len(p_tokens) + len(ra_tokens) + len(rb_tokens)+len(a_end_token) + len(b_end_token) + len(p_end_token)
    input_ids =[]
    if all_tokens_num > max_length:
        remain_tokens_num = max_length - prev_tokens_num  - 3*len(dot_tokens) 
        if remain_tokens_num >5:
            p_tokens  =  p_tokens[:int(remain_tokens_num*0.15)] + dot_tokens+ p_tokens[-int(remain_tokens_num*0.05):] if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
            ra_tokens = ra_tokens[:int(remain_tokens_num*0.3)] + dot_tokens+ ra_tokens[-int(remain_tokens_num*0.1):] if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
            rb_tokens = rb_tokens[:int(remain_tokens_num*0.3)] + dot_tokens+ rb_tokens[-int(remain_tokens_num*0.1):] if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
            input_ids = p_tokens+p_end_token + ra_tokens +a_end_token+ rb_tokens+b_end_token
        #break
    else:
        prev_tokens_num = all_tokens_num
        input_ids = p_tokens+p_end_token + ra_tokens +a_end_token+ rb_tokens+b_end_token
    input_ids += final_p_tokens
    text=tokenizer.decode(input_ids, skip_special_tokens=False)
    # text=row['examples']+text
    conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":text},
        ]
    final_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)+'Answer:\n'
    return  final_prompt 



model_path ="merge_model/qwen14b_stage1"# "merge_model/merge_qwen"#model/qwen-14b "awq_models/awq_qwen"#
if model_path in ['/kaggle/input/meta-llama-3.3-70b/transformers/ibnzterrell-instruct-awq-int4/1',
                  '/kaggle/input/qwen-72b-gptq-int4/transformers/qwen2.5-72b-instruct-gptq-int4/1']:
    print("Offload needed")
    offload = 8.5
    swap = 1
    max_len = 4000
    num_seqs = 35
else:
    offload = 0
    swap = 1
    max_len =12000#32768
    num_seqs = 128


max_token=4096

tokenizer = AutoTokenizer.from_pretrained(model_path)


df["text"] = df.apply(lambda row: preprocess_function_truncation(row, tokenizer,max_length=max_token,system_prompt=system_prompt,system_prompt_chatformat=system_prompt_chatformat),axis=1)

def tok_len(txt):
    return len(tokenizer.encode(txt))    
df['token_length'] = df['text'].apply(tok_len)
# df = df.sort_values(by='token_length', ascending=True)
print(df['token_length'])
print("max_len",max(df['token_length']))
print(df['text'][0])

print("\n\n")

llm = vllm.LLM(model=model_path,
    tensor_parallel_size= 2,
    # rope_scaling={"rope_type": "dynamic", "factor": 2.0},
    # enable_chunked_prefill=True,           
    # quantization="awq",
    # max_num_batched_tokens=8092,
    gpu_memory_utilization= 0.9, 
    trust_remote_code= True,
    dtype= "half",
    enforce_eager= True,
    max_model_len= max_len,
    disable_log_stats= True,
    cpu_offload_gb= offload,
    swap_space= swap,
    device= 'cuda',
    max_num_seqs= num_seqs,
    enable_prefix_caching= True,
)

responses = llm.generate(
        df["text"].values,
        vllm.SamplingParams(
            n=1, top_k=1, max_tokens=1, temperature=0, skip_special_tokens=False, logprobs=20,
            # logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["A","B"])]
        ),
        use_tqdm=True
    )
# print(df["text"].valu[16])
# print(df["text"].values[16])
a_tok_id = tokenizer("A", add_special_tokens=False)["input_ids"][-1]
b_tok_id = tokenizer("B", add_special_tokens=False)["input_ids"][-1]


result_a=[]
result_b=[]
n=0
for response in responses:
    # print(response.outputs[0])
    logprob_dict = response.outputs[0].logprobs[0]
    n +=1
    top_tok_ids = set(list(logprob_dict.keys()))

    if len(top_tok_ids.intersection(set([a_tok_id, b_tok_id]))) == 0:
        print(f"Bad Output for ",n)
        # if y=="model_a":
        #     result_a.append(0)
        #     result_b.append(-100)
        # else:
        #     result_a.append(-100)
        #     result_b.append(0)
        continue
    if a_tok_id in logprob_dict:
        a_logit = logprob_dict[a_tok_id].logprob
        result_a.append(a_logit )

    if b_tok_id in logprob_dict:
        b_logit = logprob_dict[b_tok_id].logprob
        result_b.append(b_logit)

df["a_logit"] = result_a
df["b_logit"] = result_b
df = df.drop(columns=['text', 'token_length'])
df.to_parquet('data/full_swap_open-model-soft_label.parquet')


