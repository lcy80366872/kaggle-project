import argparse
import os
import random
from copy import deepcopy

import kagglehub
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from train_utils import eedi_process_df


def show_distribution(df, col="label"):
    print("--" * 40)
    print(f"Number of samples: {len(df)}")
    print(df[col].value_counts())
    print("--" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = OmegaConf.load(f)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load data -----------------------------------------------------------------------------------#
    try:
        data_dir = kagglehub.competition_download(cfg.dataset.comp_dataset)
    except Exception:
        data_dir = kagglehub.dataset_download(cfg.dataset.comp_dataset)

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    content_df = pd.read_csv(os.path.join(data_dir, "misconception_mapping.csv")).rename(columns={"MisconceptionId": "content_id"})

    fold_dir = kagglehub.dataset_download(cfg.dataset.fold_dataset)
    fold_df = pd.read_parquet(os.path.join(fold_dir, "folds.parquet"))
    df = pd.merge(df, fold_df, on="QuestionId", how="left")
    df["kfold"] = df["kfold"].fillna(99)

    print("Fold distribution:")
    print(df.kfold.value_counts())
    print("--" * 40)

    if cfg.full_fit:
        train_df = df.copy()
        valid_df = df.sample(frac=0.01)
    else:
        train_df = df[df["kfold"] != cfg.fold].copy()
        valid_df = df[df["kfold"] == cfg.fold].copy()

    query_df_train, corr_df_train, content2query_train = eedi_process_df(train_df)
    query_df_valid, corr_df_valid, content2query_valid = eedi_process_df(valid_df)

    # Stage one retrieval results -----------------------------------------------------------------#
    ret_df = pd.read_parquet(cfg.dataset.stage_one_path)
    try:
        ret_df["MisconceptionId"] = ret_df["MisconceptionId"].apply(lambda x: list(map(int, x.split())))
    except Exception as e:
        print(e)
        ret_df["MisconceptionId"] = ret_df["MisconceptionId"].apply(lambda x: list(map(int, x)))

    ret_df = ret_df[["QuestionId_Answer", "MisconceptionId"]].rename(columns={"MisconceptionId": "content_id", "QuestionId_Answer": "query_id"})

    # Prepare training data -----------------------------------------------------------------------#
    ret_df_train = deepcopy(ret_df)

    # Sample negatives based on cutoffs and probabilities
    def sample_negatives(x):
        high = x[: cfg.train_sampling.high_cutoff]
        mid = x[cfg.train_sampling.high_cutoff : cfg.train_sampling.mid_cutoff]
        low = x[cfg.train_sampling.mid_cutoff : cfg.train_sampling.low_cutoff]

        high_mask = np.random.random(len(high)) < cfg.train_sampling.high_p
        mid_mask = np.random.random(len(mid)) < cfg.train_sampling.mid_p
        low_mask = np.random.random(len(low)) < cfg.train_sampling.low_p

        sampled = []
        sampled.extend([x for x, m in zip(high, high_mask) if m])
        sampled.extend([x for x, m in zip(mid, mid_mask) if m])
        sampled.extend([x for x, m in zip(low, low_mask) if m])

        sampled = list(set(sampled))

        # shuffle
        random.shuffle(sampled)

        rem = cfg.train_sampling.min_candidates - len(sampled)
        if rem > 0:
            pool = list(set(x).difference(set(sampled)))
            sampled.extend(random.sample(pool, rem))

        # cutoff
        sampled = sampled[: cfg.train_sampling.max_candidates]

        return sampled

    ret_df_train["content_id"] = ret_df_train["content_id"].apply(sample_negatives)

    flat_df_train = ret_df_train[["query_id", "content_id"]].explode(["content_id"]).reset_index(drop=True)
    flat_df_train = pd.concat([corr_df_train, flat_df_train]).reset_index(drop=True)
    flat_df_train = flat_df_train.drop_duplicates().reset_index(drop=True)

    query_df_train = query_df_train.merge(flat_df_train, on="query_id", how="left")
    query_df_train = query_df_train.merge(content_df, on="content_id", how="left")
    corr_df_train = corr_df_train.rename(columns={"content_id": "true_content_id"})
    query_df_train = query_df_train.merge(corr_df_train, on="query_id", how="left")
    query_df_train["label"] = query_df_train.apply(lambda x: int(x["content_id"] == x["true_content_id"]), axis=1)

    final_df_train = query_df_train[
        ["query_id", "content_id", "SubjectName", "ConstructName", "QuestionText", "CorrectAnswerText", "InCorrectAnswerText", "MisconceptionName", "AllOptionText", "label"]
    ].copy()

    # save
    final_df_train.to_parquet(os.path.join(cfg.output_dir, "train.parquet"))
    print("Train data distribution:")
    show_distribution(final_df_train, "label")

    print("Train # candidates distribution:")
    print(final_df_train.query_id.value_counts().value_counts().sort_index())

    # Prepare validation data ---------------------------------------------------------------------#
    if cfg.prep_valid:
        ret_df_valid = deepcopy(ret_df)

        neg_depth_valid = cfg.valid_negative_depth
        ret_df_valid["content_id"] = ret_df_valid["content_id"].apply(lambda x: x[:neg_depth_valid])
        flat_df_valid = ret_df_valid[["query_id", "content_id"]].explode(["content_id"]).reset_index(drop=True)

        # adding correct answer --
        flat_df_valid = pd.concat([corr_df_valid, flat_df_valid]).reset_index(drop=True)
        flat_df_valid = flat_df_valid.drop_duplicates().reset_index(drop=True)

        query_df_valid = query_df_valid.merge(flat_df_valid, on="query_id", how="left")
        query_df_valid = query_df_valid.merge(content_df, on="content_id", how="left")
        corr_df_valid = corr_df_valid.rename(columns={"content_id": "true_content_id"})
        query_df_valid = query_df_valid.merge(corr_df_valid, on="query_id", how="left")
        query_df_valid["label"] = query_df_valid.apply(lambda x: int(x["content_id"] == x["true_content_id"]), axis=1)

        final_df_valid = query_df_valid[
            ["query_id", "content_id", "SubjectName", "ConstructName", "QuestionText", "CorrectAnswerText", "InCorrectAnswerText", "MisconceptionName", "AllOptionText", "label"]
        ].copy()

        final_df_valid.to_parquet(os.path.join(cfg.output_dir, "valid.parquet"))
        print("Valid data distribution:")
        show_distribution(final_df_valid, "label")

    print("Done!")

    # upload to kaggle ---------------------------------------------------------------------------#
    if cfg.upload_to_kaggle:
        kagglehub.dataset_upload(cfg.upload_dataset, cfg.output_dir)

# usage: python ranker_data_prep.py --config-path ../conf/conf_ranker_prep.yaml
