import sys

import pandas as pd
import json
import pickle

from peft import PeftModel,LoraConfig,get_peft_model_state_dict, set_peft_model_state_dict,get_peft_model
from bs4 import BeautifulSoup
import numpy as np
from tqdm.auto import tqdm
from typing import Union, List, Tuple, Any
from util import *
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from torch.utils.data import Dataset
import os
from threading import Thread

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import warnings

warnings.filterwarnings('ignore')


class DatasetForReranker(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer_path: str,
            max_len: int = 512,
            query_prefix: str = 'A: ',
            passage_prefix: str = 'B: ',
            cache_dir: str = None,
            prompt: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        self.dataset = dataset
        self.max_len = max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.total_len = len(self.dataset)

        if prompt is None:
            prompt = "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
        self.prompt_inputs = self.tokenizer(prompt,
                                            return_tensors=None,
                                            add_special_tokens=False)['input_ids']
        sep = "\n"
        self.sep_inputs = self.tokenizer(sep,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        self.encode_max_length = self.max_len + len(self.sep_inputs) + len(self.prompt_inputs)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query, passage = self.dataset[item]
        # print(passage)
        query = self.query_prefix + query
        passage = self.passage_prefix + passage
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      add_special_tokens=False,
                                      max_length=self.max_len ,
                                      truncation=True)
        passage_inputs = self.tokenizer(passage,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=self.max_len,
                                        truncation=True)
        if self.tokenizer.bos_token_id is not None:
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                self.sep_inputs + passage_inputs['input_ids'],
                truncation='longest_first',
                max_length=self.encode_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
        else:
            item = self.tokenizer.prepare_for_model(
                query_inputs['input_ids'],
                self.sep_inputs + passage_inputs['input_ids'],
                truncation='longest_first',
                max_length=self.encode_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )

        item['input_ids'] = item['input_ids'] + self.sep_inputs + self.prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
        if 'position_ids' in item.keys():
            item['position_ids'] = list(range(len(item['input_ids'])))
        
        # print(item['input_ids'])
        # print(self.tokenizer.decode(item["input_ids"]))
        return item


class collater():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100
        warnings.filterwarnings("ignore",
                                message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")

    def __call__(self, data):
        labels = [feature["labels"] for feature in data] if "labels" in data[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in data:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # print("data\n",data)
        return self.tokenizer.pad(
            data,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )


def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)


def last_logit_pool_layerwise(logits: Tensor,
                              attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FlagLLMReranker:
    def __init__(
            self,
            model_name_or_path: str = None,
            lora_name_or_path: str = None,
            use_fp16: bool = False,
            use_bf16: bool = False,
            cache_dir: str = None,
            device: Union[str, int] = None
    ) -> None:

        print("llm reranker:::", model_name_or_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'mistral' in model_name_or_path.lower():
            print("left?")
            tokenizer.padding_side = 'left'

        self.tokenizer = tokenizer
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     quantization_config=bnb_config,
                                                     # cache_dir=cache_dir,
                                                     # trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                                     device_map=device)
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            bias='none',
            lora_dropout=0.05,  # Conventional
            task_type='CAUSAL_LM',#'CAUSAL_LM',
        )
        # model = get_peft_model(model, config)
        # d = torch.load(lora_name_or_path+"/adapter.bin", map_location=model.device)
        # model.load_state_dict(d, strict=False)
        model = PeftModel.from_pretrained(model, lora_name_or_path)
        for name,para in model.named_parameters():
            if '.1.' in name:
                break
            if 'lora' in name.lower():
                print(name+':')
                print('shape = ',list(para.shape),'\t','sum = ',para.sum().item())
                print('\n')
        self.model = model
        # self.model = model.merge_and_unload()
        
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.device = self.model.device
        if use_fp16 and use_bf16 is False:
            self.model.half()

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        # if device is None:
        #     self.num_gpus = torch.cuda.device_count()
        #     if self.num_gpus > 1:
        #         print(f"----------using {self.num_gpus}*GPUs----------")
        #         self.model = torch.nn.DataParallel(self.model)
        # else:
        #     self.num_gpus = 1

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 16,
                      max_length: int = 512, prompt: str = None, normalize: bool = False,
                      use_dataloader: bool = True, num_workers: int = None, disable: bool = False) -> List[float]:
        assert isinstance(sentence_pairs, list)

        # if self.num_gpus > 0:
        #     batch_size = batch_size * self.num_gpus

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        dataset, dataloader = None, None
        if use_dataloader:
            if num_workers is None:
                num_workers = min(batch_size, 16)
            dataset = DatasetForReranker(sentences_sorted,
                                         self.model_name_or_path,
                                         max_length,
                                         cache_dir=self.cache_dir,
                                         prompt=prompt)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False,
                                    num_workers=num_workers,
                                    collate_fn=collater(self.tokenizer, max_length))

        all_scores = []
        if dataloader is not None:
            for inputs in tqdm(dataloader, disable=disable):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())
        else:
            if prompt is None:
                prompt = "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
            else:
                print(prompt)
            prompt_inputs = self.tokenizer(prompt,
                                           return_tensors=None,
                                           add_special_tokens=False)['input_ids']
            sep = "\n"
            sep_inputs = self.tokenizer(sep,
                                        return_tensors=None,
                                        add_special_tokens=False)['input_ids']
            encode_max_length = max_length * 2 + len(sep_inputs) + len(prompt_inputs)
            for batch_start in trange(0, len(sentences_sorted), batch_size,desc="Processing batches"):
                batch_sentences = sentences_sorted[batch_start:batch_start + batch_size]
                batch_sentences = [(f'A: {q}', f'B: {p}') for q, p in batch_sentences]
                queries = [s[0] for s in batch_sentences]
                passages = [s[1] for s in batch_sentences]
                queries_inputs = self.tokenizer(queries,
                                                return_tensors=None,
                                                add_special_tokens=False,
                                                max_length=max_length,
                                                truncation=True)
                passages_inputs = self.tokenizer(passages,
                                                 return_tensors=None,
                                                 add_special_tokens=False,
                                                 max_length=max_length,
                                                 truncation=True)

                batch_inputs = []
                for query_inputs, passage_inputs in zip(queries_inputs['input_ids'], passages_inputs['input_ids']):
                    if self.tokenizer.bos_token_id is not None:
                        item = self.tokenizer.prepare_for_model(
                            [self.tokenizer.bos_token_id] + query_inputs,
                            sep_inputs + passage_inputs,
                            truncation='longest_first',
                            max_length=256,
                            padding=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                            add_special_tokens=False
                        )
                    else:
                        item = self.tokenizer.prepare_for_model(
                            query_inputs,
                            sep_inputs + passage_inputs,
                            truncation='longest_first',
                            max_length=256,
                            padding=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                            add_special_tokens=False
                        )
                    
                    # item = self.tokenizer.prepare_for_model(
                    #     [self.tokenizer.bos_token_id] + query_inputs,
                    #     sep_inputs + passage_inputs,
                    #     truncation='longest_first',
                    #     max_length=256,
                    #     padding=False,
                    #     return_attention_mask=False,
                    #     return_token_type_ids=False,
                    #     add_special_tokens=False
                    # )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                    if 'position_ids' in item.keys():
                        item['position_ids'] = list(range(len(item['input_ids'])))
                    batch_inputs.append(item)
                    # print(query_inputs)
                    # print(passage_inputs)
                collater_instance = collater(self.tokenizer, max_length)
                batch_inputs = collater_instance(
                    [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
                     batch_inputs])

                batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}
                # print(self.model.device)
                outputs = self.model(**batch_inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, batch_inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


def inference(df, model):
    scores = model.compute_score(list(df['predict_text'].values), batch_size=8, max_length=256,
                                 use_dataloader=True, prompt=prompt,
                                 disable=False)
    df['score'] = scores
    return df


if __name__ == '__main__':
    TOP_NUM = 5

    model_name = sys.argv[1]
    rank_model_path = sys.argv[2]
    device_num = int(sys.argv[3])
    rank_lora_path = sys.argv[4]
    misconception_mapping = pd.read_csv( 'input/misconception_mapping.csv')

    misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
    misconception_mapping['order_id'] = misconception_mapping['MisconceptionId']
    
    misconception_mapping['MisconceptionId'] = misconception_mapping['MisconceptionId'].map(lambda x: int(x))
    misconception_mapping_dic = misconception_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()
    # with open("../data/AQA-test-public/pid_to_title_abs_update_filter.json") as f:
    #     data = json.load(f)
    
    dev_df = pd.read_parquet("file/recall_50.parquet")
    print(dev_df['top_recall_pids'][0])
    # dev_df=train_df
    # dev_df=dev_df[:5]
    print(dev_df.shape)
    # actual = dev_df['answer_id'].to_list()
    # predicted = dev_df['top_recall_pids'].to_list()
    # train_map_score = mapk(actual, predicted, k=25)
    # print(f'mapk score = {train_map_score}')
    
    
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    
    def recall_context(x, misconception_mapping_dic):
        res = []
        for xi in x:
            MisconceptionName = misconception_mapping_dic[xi]
            res.append(MisconceptionName)
        return res
    
    dev_df['top_recall_text'] = dev_df['top_recall_pids'].apply(lambda x: recall_context(x, misconception_mapping_dic))
    
    dev_df['order_id'] = list(range(len(dev_df)))
    predict_list = []
    for _, row in dev_df.iterrows():
        for index, text in enumerate(row['top_recall_text'][:TOP_NUM]):
            predict_list.append({
                "order_id": row['order_id'],
                'predict_text': [row['query_text'], text],
                'or_order': index,
                'candi_id': row['top_recall_pids'][index]
            })
    predict_list = pd.DataFrame(predict_list)
    print(predict_list.shape)
    
    prompt = "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
    
    use_device = [f'cuda:{i}' for i in range(device_num)]
    print(use_device)
    models = []
    for device in use_device:
        print(device)
        reranker = FlagLLMReranker(
            rank_model_path,
            lora_name_or_path=rank_lora_path,
            use_bf16=True, device=device)
        models.append(reranker)
    
    # infer
    results = {}
    
    def run_inference(df, model, index):
        results[index] = inference(df, model)
    
    ts = []
    
    predict_list['fold'] = list(range(len(predict_list)))
    predict_list['fold'] = predict_list['fold'] % len(use_device)
    print(predict_list['fold'].value_counts())
    for index, device in enumerate(use_device):
        t0 = Thread(target=run_inference, args=(predict_list[predict_list['fold'] == index], models[index], index))
        ts.append(t0)
    
    for i in range(len(ts)):
        ts[i].start()
    for i in range(len(ts)):
        ts[i].join()
    
    predict_list = pd.concat(list(results.values()), axis=0)
    
    predicts_test = []
    for _, df in predict_list.groupby("order_id"):
        scores = df['score'].values
        score_indexs = np.argsort(-scores)
        candi_ids = df['candi_id'].values
        predicts_test.append([candi_ids[index] for index in score_indexs[:25]])
    result = []
    # 遍历A和B的每个对应位置元素
    for a, b in zip(predicts_test, dev_df['top_recall_pids'].tolist()):
        # 获取b数组从第6个元素开始的部分（索引5及以后的元素）
        b_from_fifth = b[TOP_NUM :]
        # 将a和b_from_fifth拼接
        result.append(np.concatenate((a, b_from_fifth)))
    
    # dev_df["MisconceptionId"] = predicts_test
    dev_df['MisconceptionId'] = [' '.join(map(str,c)) for c in result]
    sub = []
    for _,row in dev_df.iterrows():
        sub.append(
            {
                "QuestionId_Answer":f"{row['QuestionId']}_{row['answer_name']}",
                "MisconceptionId":row['MisconceptionId']
            }
        )
    submission_df = pd.DataFrame(sub)
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")
    
    
    # dev_df.to_csv("submission.csv", columns=["QuestionId_Answer", "MisconceptionId"], index=False)
    # actual = dev_df['answer_id'].to_list()
    # predicted = predicts_test
    # train_map_score = mapk(actual, predicted, k=25)
    # print(f'mapk score = {train_map_score}')
    actual = dev_df['answer_id'].to_list()
    predicted = result
    train_map_score = mapk(actual, predicted, k=25)
    print(f'mapk score = {train_map_score}')