from guidance.llms import Transformers

# from transformers import LlamaForCausalLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# from optimum.bettertransformer import BetterTransformer
import os
import sys
import glob
import torch

from model_roles import Llama2ChatRole, Llama2GuanacoRole

selected_role = None


def get_role():
    return selected_role
    # return Vicuna1_3Role
    # return Llama2ChatRole
    # return Llama2GuanacoRole
    # return Llama2UncensoredChatRole


class LLaMATransformer(Transformers):
    """A HuggingFace transformers version of the LLaMA language model with Guidance support."""

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAAutoGPTQ"

        global selected_role
        if "guanaco" in model.lower():
            print("found a Guanaco model")
            selected_role = Llama2GuanacoRole
        elif "llama-2-7b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole
        elif "llama-2-13b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole

        print(f"Initializing LLaMAAutoGPTQ with model {model}")

        low_vram_mode = "--low-vram" in sys.argv
        if low_vram_mode:
            print("low vram mode enabled")

        tokenizer = AutoTokenizer.from_pretrained(model)

        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=low_vram_mode,
            bnb_4bit_quant_type="nf4",
        )

        config = AutoConfig.from_pretrained(model)
        config.pretraining_tp = 1
        model = AutoModelForCausalLM.from_pretrained(
            model,
            config=config,
            quantization_config=double_quant_config,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )

        # model = BetterTransformer.transform(model, keep_original_model=False)

        # model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        return get_role().role_start(role)

    @staticmethod
    def role_end(role):
        return get_role().role_end(role)


def find_safetensor_filename(dir):
    # Make sure the directory path ends with '/'
    if dir[-1] != "/":
        dir += "/"

    # Use glob to find all files with the given extension
    files = glob.glob(dir + "*.safetensors")

    # If there is at least one file, return the first one
    if len(files) == 0:
        print(f"Error: no safetensor file found in {dir}")
        return None
    elif len(files) == 1:
        return os.path.basename(files[0])
    else:
        print(f"Warning: multiple safetensor files found in {dir}, picking just one")
        return os.path.basename(files[0])
