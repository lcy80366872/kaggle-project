from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import gc
import pandas as pd
import pickle
import sys
import numpy as np
from tqdm.autonotebook import trange
from sklearn.model_selection import GroupKFold
import json
import torch
from numpy.linalg import norm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel,BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
import json
import copy
import warnings
import os
warnings.filterwarnings('ignore')
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
def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

# def last_token_pool(last_hidden_states: Tensor,
#                     attention_mask: Tensor) -> Tensor:
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# def get_detailed_instruct(task_description: str, query: str) -> str:
#     return f'Instruct: {task_description}\nQuery: {query}'

# def get_predict(df, query_embedding, sentence_embedding, index_paper_text_embeddings_index, device, RANK=100):
#     predict_list = []
#     for i, (_, row) in enumerate(df.iterrows()):
#         query_id = row['order_index']
#         query_em = query_embedding[query_id].reshape(1, -1)
#         query_em = torch.tensor(query_em).to(device).view(1, -1)
        
#         score = (query_em @ sentence_embedding.T).squeeze()
        
#         sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:RANK]
#         pids = [index_paper_text_embeddings_index[index] for index in sort_index]
#         predict_list.append(pids)
#     return predict_list

# def inference(df, model, tokenizer, device):
#     batch_size=16
#     max_length=512
#     pids = list(df['order_index'].values)
#     sentences = list(df['query_text'].values)
    
#     # 根据文本长度逆序：长的在前，短的在后
#     length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
#     sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
#     # print(length_sorted_idx[:5])
#     # print(sentences_sorted[:5])
    
#     all_embeddings = []
#     for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=False):
#         sentences_batch = sentences_sorted[start_index: start_index + batch_size]
#         features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
#         # if start_index == 0:
#         #     print(f'features = {features.keys()}')
#         # features input_id, attention_mask
#         features = batch_to_device(features, device)
#         with torch.no_grad():
#             outputs = model(**features)
#             embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
#             # if start_index == 0:
#             #     print(f'embeddings = {embeddings.detach().cpu().numpy().shape}')
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#             embeddings = embeddings.detach().cpu().numpy().tolist()
#         all_embeddings.extend(embeddings)
    
#     all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]
    
#     sentence_embeddings = np.concatenate(all_embeddings, axis=0)
#     result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
#     return result

# path_prefix = "input"
# # model_path = "/kaggle/input/sfr-embedding-mistral/SFR-Embedding-2_R"
# model_path = "model/qwen-14b"

# lora_path="model_save/qwen2_round7_qlora_rerun/epoch_9_model/adapter.bin"
# # lora_path=None
# device='cuda:0'
# VALID = False


# from typing import List, Tuple, Optional, Union
# from transformers import set_seed, AutoConfig, AutoModel, MistralPreTrainedModel, MistralConfig, DynamicCache, Cache, \
#     Qwen2PreTrainedModel, Qwen2Config
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
#     _prepare_4d_causal_attention_mask
# from transformers.modeling_outputs import BaseModelOutputWithPast
# # from transformers.models.mistral.modeling_flax_mistral import MISTRAL_INPUTS_DOCSTRING
# from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
# from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm
# from torch import nn, Tensor
# class Qwen2Model(Qwen2PreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

#     Args:
#         config: Qwen2Config
#     """

#     def __init__(self, config: Qwen2Config):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self._attn_implementation = config._attn_implementation
#         self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#         self.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             labels: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[List[torch.FloatTensor]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         past_key_values_length = 0

#         if use_cache:
#             use_legacy_cache = not isinstance(past_key_values, Cache)
#             if use_legacy_cache:
#                 past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#             past_key_values_length = past_key_values.get_usable_length(seq_length)

#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
#             is_padding_right = attention_mask[:, -1].sum().item() != batch_size
#             if is_padding_right:
#                 raise ValueError(
#                     "You are attempting to perform batched generation with padding_side='right'"
#                     " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
#                     " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
#                 )

#         if self._attn_implementation == "flash_attention_2":
#             # 2d mask is passed through the layers
#             attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
#         elif self._attn_implementation == "sdpa" and not output_attentions:
#             # output_attentions=True can not be supported when using SDPA, and we fall back on
#             # the manual implementation that requires a 4D causal mask in all cases.
#             attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
#                 attention_mask,
#                 (batch_size, seq_length),
#                 inputs_embeds,
#                 past_key_values_length,
#             )
#         else:
#             # 4d mask is passed through the layers
#             attention_mask = _prepare_4d_causal_attention_mask(
#                 attention_mask,
#                 (batch_size, seq_length),
#                 inputs_embeds,
#                 past_key_values_length,
#                 sliding_window=self.config.sliding_window,
#             )

#         hidden_states = inputs_embeds

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = None
#         if use_cache:
#             next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

# device_0 = torch.device('cuda:0')
# device_1 = torch.device('cuda:1')

# if lora_path:
#     tokenizer_0 = AutoTokenizer.from_pretrained(lora_path.replace("/adapter.bin",""))
#     tokenizer_1 = AutoTokenizer.from_pretrained(lora_path.replace("/adapter.bin",""))
# else:
#     tokenizer_0 = AutoTokenizer.from_pretrained(model_path)
#     tokenizer_1 = AutoTokenizer.from_pretrained(model_path)
# print(lora_path)
# bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
# model_0 = Qwen2Model.from_pretrained(model_path, 
#                                   quantization_config=bnb_config, 
#                                   device_map=device_0)
#                                 #  trust_remote_code=True)
# model_1 = Qwen2Model.from_pretrained(model_path, 
#                                   quantization_config=bnb_config, 
#                                   device_map=device_1)
#                                 #  trust_remote_code=True)

# if lora_path:
#     print("loading lora")
#     config = LoraConfig(
#         r=64,
#         lora_alpha=128,
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#         ],
#         bias="none",
#         lora_dropout=0.05,  # Conventional
#         task_type="FEATURE_EXTRACTION",#FEATURE_EXTRACTION
#     )
#     model_0 = get_peft_model(model_0, config)
#     d_0 = torch.load(lora_path, map_location=model_0.device)
#     model_0.load_state_dict(d_0, strict=False)
#     model_1 = get_peft_model(model_1, config)
#     d_1 = torch.load(lora_path, map_location=model_1.device)
#     model_1.load_state_dict(d_1, strict=False)
#     # model = model.merge_and_unload()
# model_1 = model_1.eval()
# model_0 = model_0.eval()
# for name,para in model_0.named_parameters():
#     if '.1.' in name:
#         break
#     if 'lora' in name.lower():
#         print(name+':')
#         print('shape = ',list(para.shape),'\t','sum = ',para.sum().item())
#         print('\n')
# task_description = 'Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.'
# tra = pd.read_csv(f"{path_prefix}/train.csv")
# tra=tra[tra['QuestionId'] % 5 == 0].reset_index(drop=True)
# print(tra.shape)
# misconception_mapping = pd.read_csv(f"{path_prefix}/misconception_mapping.csv")

# train_data = []
# for _,row in tra.iterrows():
#     for c in ['A','B','C','D']:
#         if c ==row['CorrectAnswer']:
#             continue
#         if f'Answer{c}Text' not in row:
#             continue
#         if str(row[f"Misconception{c}Id"])=="nan":
#             continue
#         real_answer_id = row['CorrectAnswer']
#         real_text = row[f'Answer{real_answer_id}Text']
#         query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{row[f'Answer{c}Text']}"
#         row['query_text'] = get_detailed_instruct(task_description,query_text)
#         row['answer_name'] = c
#         row['answer_id'] =  [int(row[f'Misconception{c}Id'])]
#         train_data.append(copy.deepcopy(row))
# train_df = pd.DataFrame(train_data)
# train_df["length"] = train_df["query_text"].apply(len)
# train_df = train_df.sort_values("length", ascending=False)
# train_df['order_index'] = list(range(len(train_df)))

# # train_embeddings = inference(train_df, model_0, tokenizer_0, device_0)

# sub_1 = train_df.iloc[0::2].copy()
# sub_2 = train_df.iloc[1::2].copy()
# import copy
# from concurrent.futures import ThreadPoolExecutor
# with ThreadPoolExecutor(max_workers=2) as executor:
#     results = executor.map(inference, (sub_1, sub_2), (model_0, model_1),(tokenizer_0, tokenizer_1), (device_0, device_1))
# results_list = list(results)
# result1=copy.deepcopy(results_list[0])
# result2=copy.deepcopy(results_list[1])
# result1.update(result2)

# train_embeddings=result1



# misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
# misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']
# misconception_mapping['MisconceptionId'] = misconception_mapping['MisconceptionId'].map(lambda x: int(x))
# misconception_mapping_dic = misconception_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()
# misconception_mapping["length"] = misconception_mapping["query_text"].apply(len)
# misconception_mapping = misconception_mapping.sort_values("length", ascending=False)



# # the total #tokens in sub_1 and sub_2 should be more or less the same
# mis_sub_1 = misconception_mapping.iloc[0::2].copy()
# mis_sub_2 = misconception_mapping.iloc[1::2].copy()
# with ThreadPoolExecutor(max_workers=2) as executor:
#     mis_results = executor.map(inference, (mis_sub_1, mis_sub_2), (model_0, model_1),(tokenizer_0, tokenizer_1), (device_0, device_1))
# results_list = list(mis_results)
# result1=copy.deepcopy(results_list[0])
# result2=copy.deepcopy(results_list[1])
# result1.update(result2)
# doc_embeddings =result1
# # doc_embeddings = inference(misconception_mapping, model_0, tokenizer_0, device_0)
# # print(doc_embeddings )
# sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(doc_embeddings.values())])
# sentence_embeddings_tensor = torch.tensor(sentence_embeddings).to(device_0)
# index_paper_text_embeddings_index = {index: paper_id for index, paper_id in enumerate(list(doc_embeddings.keys()))}





# predict_train = get_predict(train_df, train_embeddings, sentence_embeddings_tensor, index_paper_text_embeddings_index, device_0, RANK=25)

# train_df['top_recall_pids']=predict_train
# train_df.to_parquet("file/recall_50.parquet", index=False)

# # from numba import cuda
# # device = cuda.get_current_device()
# # device.reset()
# import gc

# del  model_0, tokenizer_0, model_1, tokenizer_1

# gc.collect()
# torch.cuda.empty_cache()


import sys

import pandas as pd
import json
import pickle

from peft import PeftModel
# from bs4 import BeautifulSoup
import numpy as np
from tqdm.auto import tqdm
from typing import Union, List, Tuple, Any

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
                truncation='only_second',
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
                truncation='only_second',
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
        # print(data)
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
                                                     cache_dir=cache_dir,
                                                     # trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                                     device_map=device)

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
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
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
            for batch_start in trange(0, len(sentences_sorted), batch_size):
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
                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs,
                        sep_inputs + passage_inputs,
                        truncation='only_second',
                        max_length=256,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
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
    scores = model.compute_score(list(df['predict_text'].values), batch_size=2, max_length=512,
                                 use_dataloader=True, prompt=prompt,
                                 disable=False)
    df['score'] = scores
    return df

TOP_NUM = 5

rank_model_path = "model/qwen-14b"
device_num = 2
rank_lora_path = "rank_save/qwen2.5_14b_round7_qlora_rerun"
misconception_mapping = pd.read_csv( 'input/misconception_mapping.csv')

misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_id'] = misconception_mapping['MisconceptionId']

misconception_mapping['MisconceptionId'] = misconception_mapping['MisconceptionId'].map(lambda x: int(x))
misconception_mapping_dic = misconception_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()
# with open("../data/AQA-test-public/pid_to_title_abs_update_filter.json") as f:
#     data = json.load(f)

dev_df = pd.read_parquet("file/recall_50.parquet")
# dev_df=train_df
# dev_df=dev_df[:5]
print(dev_df.shape)
actual = dev_df['answer_id'].to_list()
predicted = dev_df['top_recall_pids'].to_list()
print(actual[:5])
# print(dev_df['top_recall_pids'].to_list())
print(predicted[:5])
train_map_score = mapk(actual, predicted, k=25)
print(f'mapk score = {train_map_score}')


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
        use_fp16=False, device=device)
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
submission_df.to_csv("file/submission.csv", index=False)
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