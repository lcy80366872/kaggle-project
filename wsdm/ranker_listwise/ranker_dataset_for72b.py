from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm 
system_prompt_v0='''You are a skilled judge evaluating responses from two large language models(LLMs). Your task is to select the response that best meets the user's needs based on the query provided.

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

**Output:**
Respond with only a single letter:
- A if <Response_A> is better.
- B if <Response_B> is better.

**Important Notes:**
- Provide only the letter A or B as your response.
- No explanations are needed.

**Example:**
Input:

<Query>
What is the capital of France?
</Query>

<Response_A>
The capital of France is Paris.
</Response_A>

<Response_B>
Paris is the capital of France. It's a beautiful city with lots of history.
</Response_B>

Which response is more likely to be selected by a user? (A or B)
Output:
A'''

system_prompt_chatformat_v0='''system
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

**Output:**
Respond with only a single letter:
- A if <Response_A> is better.
- B if <Response_B> is better.

**Important Notes:**
- Provide only the letter A or B as your response.
- No explanations are needed.

**Example:**
Input:

<Query>
What is the capital of France?
</Query>

<Response_A>
The capital of France is Paris.
</Response_A>

<Response_B>
Paris is the capital of France. It's a beautiful city with lots of history.
</Response_B>

Which response is more likely to be selected by a user? (A or B)
Output:
A'''
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
def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=False,#cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )
    tokenizer.add_bos_token = True
    tokenizer.padding_side = "left"  # use left padding

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.bos_token =  "<|im_start|>"
    tokenizer.bos_token_id = 151644
    print( tokenizer.bos_token)
    print(tokenizer.eos_token)
    print( tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)
    return tokenizer


class RankerDataset:
    """
    Dataset class for EEDI - Misconception Detection
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["text"]+"Answer:\n",
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        expl_model_inputs =self.tokenizer(
            examples["text"]+"Thought and answer:\n" ,
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        cot_text=examples['cot_text'] if examples['cot_text'] else "No thought."
        cot="Thought:"+cot_text+"\nAnswer:\n"+examples["winner"].replace("model_","").upper()
        rationale_output_encodings = self.tokenizer(cot, max_length=self.cfg.model.max_length, truncation=True,padding=False, return_length=True)
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
        return model_inputs
    def preprocess_function_truncation(self, df,is_train=False, ad_fs=False):
        process_text = []
        dot_tokens = self.tokenizer("......", add_special_tokens=False)["input_ids"]
        system_token= [self.tokenizer.bos_token_id]+self.tokenizer(system_prompt_chatformat,add_special_tokens=False) ["input_ids"]+[self.tokenizer.eos_token_id]
        final_p_tokens = self.tokenizer('Which response is more likely to be selected by a user? (A or B)\n', add_special_tokens=False)["input_ids"]
        user_token=self.tokenizer("user\n", add_special_tokens=False)["input_ids"]
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data truncation"): 
            if ad_fs:
                max_length= self.cfg.model.max_length-row['examples_tokens_len']
            else:
                max_length=self.cfg.model.max_length
            p=row['prompt']
            ra= row['response_a']
            rb =row['response_b']
            prev_tokens_num =  len(system_token)+len(final_p_tokens) +len(user_token)
            prompt=f'''**Here is your input to process now-**\nInput:\n\n<Query>\n{p}'''
            response_a=f'''\n<Response_A>\n{ra}'''
            response_b=f'''\n<Response_B>\n{rb}'''
            p_tokens  = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ra_tokens = self.tokenizer(response_a, add_special_tokens=False)["input_ids"]

            rb_tokens = self.tokenizer(response_b, add_special_tokens=False)["input_ids"]
            a_end_token=self.tokenizer(f"""\n</Response_A>\n{'---'*10}\n""", add_special_tokens=False)["input_ids"]
            b_end_token=self.tokenizer(f"""\n</Response_B>\n\n""", add_special_tokens=False)["input_ids"]
            p_end_token=self.tokenizer(f"""\n</Query>\n{'---'*10}\n""", add_special_tokens=False)["input_ids"]
            all_tokens_num = prev_tokens_num +  len(p_tokens) + len(ra_tokens) + len(rb_tokens)+len(a_end_token) + len(b_end_token) + len(p_end_token)
            input_ids=[]
            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num  - 3*len(dot_tokens) 
                if remain_tokens_num >5:
                    p_tokens  =  p_tokens[:int(remain_tokens_num*0.15)] + dot_tokens+ p_tokens[-int(remain_tokens_num*0.05):] if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.3)] + dot_tokens+ ra_tokens[-int(remain_tokens_num*0.1):] if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.3)] + dot_tokens+ rb_tokens[-int(remain_tokens_num*0.1):] if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                    input_ids = p_tokens+p_end_token + ra_tokens +a_end_token+ rb_tokens+b_end_token
            else:
                prev_tokens_num = all_tokens_num
                input_ids = p_tokens+p_end_token + ra_tokens +a_end_token+ rb_tokens+b_end_token

            input_ids += final_p_tokens
            text=self.tokenizer.decode(input_ids, skip_special_tokens=False)
            if ad_fs:
                text=row['examples']+text
            conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":text},
                ]
            final_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            process_text.append(final_prompt)
        df["text"] = process_text
        return df

    def preprocess_function(self, df, is_train=False, rng=None):
        formatted_texts = []
        system = system_prompt

        for _, row in df.iterrows():
            # few_shot_examples = row["examples"]

            user_message = ""
            # if len(few_shot_examples.strip()) > 0:
            #     user_message += "The following reference examples are provided to help you understand relevant responses and the perfering choice.\n"
            #     user_message += f"Reference examples:\n{few_shot_examples}\n\n"



            user_message += f'''Here is your input to process now-
Input:

<Query>
{row['prompt']}
</Query>
{'---'*10}
<Response_A>
{row['response_a']}
</Response_A>
{'---'*10}
<Response_B>
{row['response_b']}
</Response_B>

'''
            # if is_train:
            #     num_thoughts = rng.choice([0, 1, 2, 3])
            #     if num_thoughts > 0:
            #         cot_list = [row["cot_7b"], row["cot_14b"], row["cot_32b"]]
            #         selected_cots = rng.sample(cot_list, k=num_thoughts)
            #         for cot in selected_cots:
            #             user_message += f"Thought: {cot}\n\n"

            # else:
            #     user_message += f"Thought: {row['cot_7b']}\n\n"
            #     user_message += f"Thought: {row['cot_14b']}\n\n"
            #     user_message += f"Thought: {row['cot_32b']}\n\n"
            user_message += 'Which response is more likely to be selected by a user? (A or B)\nOutput:\n'



            conversation = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]

            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)

        df["text"] = formatted_texts
        return df

    def get_dataset(self, df, is_train=False, rng=None):
        """use this function to get the dataset

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Dataset: HF Dataset object with tokenized inputs and labels
        """

        df = deepcopy(df)
        # df = self.preprocess_function(df, is_train, rng)
        df = self.preprocess_function_truncation(df, is_train,ad_fs=self.cfg.model.add_fs)
        # text_column_df = df[['text']]
    
        # # 将 DataFrame 保存为 CSV 文件
        # text_column_df.to_csv("output/all_calib_data.csv", index=False)
        # text_column_df.to_parquet("output/all_calib_data.parquet", index=False)
        task_dataset = Dataset.from_pandas(df)
        # print(df.columns)
        remove_columns = [col for col in df.columns if col not in ["query_id", "content_ids", "combined_id", "winner", "teacher_logits","cot_text"]]

        task_dataset = task_dataset.map(self.tokenize_function, batched=False, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)
        # task_dataset = task_dataset.map(lambda x: self.tokenize_function(x, ad_fs= self.cfg.add_fs), batched=False, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)

        return task_dataset
