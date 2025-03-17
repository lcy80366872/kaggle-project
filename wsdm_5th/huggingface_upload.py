from huggingface_hub import HfApi

# 设置 Hugging Face 的访问令牌
api = HfApi()
token ="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"  # 替换为您的 Token # "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"#

# 设置模型仓库路径
repo_id = "RandomForest1024/V24"  # 例如：myusername/my-finetuned-model
local_dir = "merge_model"  # 替换为您的模型本地路径

# 上传模型文件夹到 Hugging Face
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    token=token,
)

print("模型上传成功！")