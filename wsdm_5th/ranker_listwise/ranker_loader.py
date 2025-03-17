from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


# class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
#     def __call__(self, features, return_tensors=None):
#         features_df = pd.DataFrame(features)
#         pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
#         expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
#             columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')

#         pred_features = super().__call__(pred_features, return_tensors)
#         expl_features = super().__call__(expl_features, return_tensors)

#         return {
#             'pred': pred_features,
#             'expl': expl_features,
#         }

@dataclass
class RankerCollator(DataCollatorWithPadding):
    """
    Data collector
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        # unlabelled_qid_th = 2000
        # query_ids = [feature["query_id"] for feature in features]
        # query_ids = [int(q.split("_")[0]) for q in query_ids]
        # ce_mask = [0.0 if qid >= unlabelled_qid_th else 1.0 for qid in query_ids]
        ce_mask = [ 1.0] *len(features)
        labels = None
        # print("features[0].keys()",features[0].keys())
        # print(features)
        if "winner" in features[0].keys():
            label_map = {"model_a": 0, "model_b": 1}
            labels = [int(label_map[feature['winner']]) for feature in features]
            # labels = [feature["label"] for feature in features]
        aux_label = [feature['aux_labels'] for feature in features]
        teacher_scores = None
        if "teacher_logits" in features[0].keys():
            teacher_scores = [feature["teacher_logits"] for feature in features]
        # for feature in features:
        #     print('aux_label',len(feature['aux_labels']))
        #     print('input_ids',len(feature['input_ids']))
        #     print('attention_mask',len(feature['attention_mask']))
        #     print('aux-input_ids',len(feature['expl_input_ids']))
        #     print('aux_attention_mask',len(feature['expl_attention_mask']))
        #     print("\n\n")
        # print(len(labels))
        features_ori = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                # "expl_input_ids": feature['expl_input_ids'],
                # "expl_attention_mask": feature["expl_attention_mask"],
            }
            for feature in features
        ]
        features_aux = [
            {
                # "input_ids": feature["input_ids"],
                # "attention_mask": feature["attention_mask"],
                "input_ids": feature['expl_input_ids'],
                "attention_mask": feature["expl_attention_mask"],
                # 'aux_labels':feature['aux_labels'],
            }
            for feature in features
        ]
        
        # features_label = [
        #     {
        #         # "input_ids": feature["input_ids"],
        #         # "attention_mask": feature["attention_mask"],
        #         "input_ids": feature['aux_labels'],
        #         # "attention_mask": feature["expl_attention_mask"],
        #         # 'aux_labels':feature['aux_labels'],
        #     }
        #     for feature in features
        # ]


        batch = self.tokenizer.pad(
            features_ori,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch_aux = self.tokenizer.pad(
            features_aux,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # batch_label = self.tokenizer.pad(
        #     features_label,
        #     padding="longest",
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        
        batch["ce_mask"] = torch.tensor(ce_mask, dtype=torch.float32)

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if teacher_scores is not None:
            batch["teacher_logits"] = torch.tensor(teacher_scores, dtype=torch.float32)
        batch["expl_input_ids"]=batch_aux["input_ids"]
        batch["expl_attention_mask"]= batch_aux["attention_mask"]

        seq_len = batch["expl_input_ids"].size(1)

        if  aux_label is not None:
            padded_labels = []
            for label in  aux_label:
                padded_label = [-100] * (seq_len - len(label)) + label  # left pad
                padded_labels.append(padded_label)
            batch["aux_labels"] = torch.tensor(padded_labels, dtype=torch.int64)
        # print("输入:", self.tokenizer.decode(batch_aux["input_ids"][0]))  
        # print("标签:", self.tokenizer.decode( batch["aux_labels"][0][ batch["aux_labels"][0] != -100])) 
        # batch['aux_labels'] = batch_label["input_ids"]
        return batch


def show_batch(batch, tokenizer, print_fn=print, **kwargs):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")
    print_fn(f"shape of attention_mask: {batch['attention_mask'].shape}")

    if "labels" in batch.keys():
        print_fn(f"shape of labels: {batch['labels'].shape}")
        print_fn(f"labels: {batch['labels']}")
        # print_fn(f"cot_labels: {batch['aux_labels'][0]}")

    print_fn("\n\n")
    for idx in range(bs):
        print_fn(f"=== Example {idx} ===")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        # print_fn(f"cot_labels:\n\n{tokenizer.decode(batch['aux_labels'][idx], skip_special_tokens=True)}")
        print_fn("~~" * 40)
