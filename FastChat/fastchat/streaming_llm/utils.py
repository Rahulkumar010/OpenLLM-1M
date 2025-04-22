import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json

from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from typing import Optional, List

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# fix the imports
def fixed_get_imports(filename: Optional[str]=None) -> List[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


def load_model(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    # create model
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto' if torch.cuda.is_available() else None,
            # device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            # torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict
