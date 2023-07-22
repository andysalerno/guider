from guidance.llms import Transformers
from transformers import LlamaForCausalLM
from huggingface_hub import snapshot_download
import os
import glob
import sys
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

def get_role():
    # return Vicuna1_3Role
    return Llama2ChatRole

class LLaMAAutoGPTQ(Transformers):
    """A HuggingFace transformers version of the LLaMA language model with Guidance support."""

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAAutoGPTQ"

        print(f"Initializing LLaMAAutoGPTQ with model {model}")

        branch = None
        if ":" in model:
            splits = model.split(":")
            model = splits[0]
            branch = splits[1]

        models_dir = "./models"
        name_suffix = model.split("/")[1]
        model_dir = f"{models_dir}/{name_suffix}"
        snapshot_download(repo_id=model, revision=branch, local_dir=model_dir)

        model_basename = find_safetensor_filename(model_dir)
        model_basename = model_basename.split(".safetensors")[0]

        print(f"found model with basename {model_basename} in dir {model_dir}")

        use_triton = True
        low_vram_mode = "--low-vram" in sys.argv

        if low_vram_mode:
            print("low vram mode enabled")

        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        final_path = f"{model_dir}"

        model = AutoGPTQForCausalLM.from_quantized(
            # model,
            final_path,
            model_basename=model_basename,
            use_safetensors=True,
            inject_fused_mlp=low_vram_mode is False,
            inject_fused_attention=low_vram_mode is False,
            device="cuda:0",
            use_triton=use_triton,
            warmup_triton=use_triton,
            quantize_config=None,
            # max_new_tokens=4096,
            # max_tokens=4096,
            # max_length=4096
        )

        # print(f'tokenizer max_new_tokens: {tokenizer.max_new_tokens}')

        model._update_model_kwargs_for_generation = (
            LlamaForCausalLM._update_model_kwargs_for_generation
        )

        # model.config.max_context = 4096
        # model.config.max_length = 4096
        # model.config.max_new_tokens = 4096

        model.config.max_seq_len = 4096 # this is the one

        # print(f'model max_new_tokens: {model.max_new_tokens}')

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


class Vicuna1_3Role:

    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""

class RedmondPufferRole:

    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""

class Llama2ChatRole:

    @staticmethod
    def role_start(role):
        if role == "user":
            # return " <s>[INST] "
            return " [INST] "
        elif role == "assistant":
            return ""
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return " [/INST] "
        elif role == "assistant":
            # return " </s>"
            return ""
        else:
            return ""