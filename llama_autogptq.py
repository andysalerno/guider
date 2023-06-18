from pathlib import Path
from guidance.llms import Transformers 
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import transformers
from gptq.utils import find_layers
from gptq import quant
import sys
from huggingface_hub import snapshot_download
import os
import glob

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

class LLaMAAutoGPTQ(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """

    llm_name: str = "llama"
    
    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAAutoGPTQ"

        print(f'Initializing LLaMAAutoGPTQ with model {model}')

        models_dir = './models'
        name_suffix = model.split('/')[1]
        model_dir = f'{models_dir}/{name_suffix}'
        snapshot_download(repo_id=model, local_dir=model_dir)

        # model_name_or_path = "TheBloke/tulu-13B-GPTQ"
        model_basename = "gptq_model-4bit-128g"

        model_basename = find_safetensor_filename(model_dir)
        model_basename = model_basename.split('.safetensors')[0]

        print(f'found model with basename {model_basename} in dir {model_dir}')

        use_triton = True

        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(model,
                model_basename=model_basename,
                use_safetensors=True,
                # device="cuda:0",
                use_triton=use_triton,
                warmup_triton=use_triton,
                quantize_config=None)
        
        model._update_model_kwargs_for_generation = LlamaForCausalLM._update_model_kwargs_for_generation
            
        return super()._model_and_tokenizer(model, tokenizer, **kwargs)


    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''
    
    @staticmethod
    def role_end(role):
        if role == 'user':
            return ''
        elif role == 'assistant':
            return '</s>'
        else:
            return ''

def find_safetensor_filename(dir):
    # Make sure the directory path ends with '/'
    if dir[-1] != "/":
        dir += "/"

    # Use glob to find all files with the given extension
    files = glob.glob(dir + "*.safetensors")

    # If there is at least one file, return the first one
    if len(files) == 0:
        print(f'Error: no safetensor file found in {dir}')
        return None
    elif len(files) == 1:
        return os.path.basename(files[0])
    else:
        print(f'Warning: multiple safetensor files found in {dir}, picking just one')
        return os.path.basename(files[0])

    # If no file was found, return None
    return None