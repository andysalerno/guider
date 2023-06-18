from pathlib import Path
from guidance.llms import Transformers 
import sys
from huggingface_hub import snapshot_download

from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from transformers import LlamaTokenizer

class ExLLaMA(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """
    
    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        # load the LLaMA specific tokenizer and model

        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAGPTQ"

        print(f'Initializing LLaMAGPTQ with model {model}')

        models_dir = './models'
        name_suffix = model.split('/')[1]
        model_dir = f'{models_dir}/{name_suffix}'
        snapshot_download(repo_id=model, local_dir=model_dir)

        (model, tokenizer) = _load(model_dir)

        print(f'Loading tokenizer from: {model_dir}')

        tokenizer = LlamaTokenizer.from_pretrained(Path(model_dir))
            
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
    

def _load(model_dir: str):
    model_dir = Path(model_dir)
    tokenizer_model_path = model_dir / "tokenizer.model"
    model_config_path = model_dir / "config.json"

    # Find the model checkpoint
    model_path = None
    for ext in ['.safetensors', '.pt', '.bin']:
        found = list(model_dir.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) > 1:
                print(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')
                sys.exit()

            model_path = found[-1]
            break

    config = ExLlamaConfig(str(model_config_path))
    config.model_path = str(model_path)
    model = ExLlama(config)
    tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
    cache = ExLlamaCache(model)

    print('Loaded model and tokenizer.')

    return ExLlamaWrapper(model), ExLlamaTokenizerWrapper(tokenizer)


class ExLlamaWrapper:
    device = None

    def __init__(self, exllama: ExLlama):
        self.exllama = exllama
    
    def to(self, device):
        pass

class ExLlamaTokenizerWrapper:
    vocab_size = 32000

    def __init__(self, tokenizer: ExLlamaTokenizer):
        self.tokenizer = tokenizer