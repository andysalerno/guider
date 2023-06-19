from pathlib import Path
from guidance.llms import Transformers 
import sys
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path("exllama")))

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
    generator = ExLlamaGenerator(model, tokenizer, cache)
    # generator.

    print('Loaded model and tokenizer.')

    return ExLlamaWrapper(model, tokenizer, cache), ExLlamaTokenizerWrapper(tokenizer)


class ExLlamaWrapper:
    device = None

    def __init__(self, model: ExLlama, tokenizer: ExLlamaTokenizer, cache: ExLlamaCache):
        self.exllama_model = model
        self.exllama_tokenizer = tokenizer
        self.exllama_cache = cache

        self.config = model.config
    
    def to(self, device):
        pass

    def generate(self, **kwargs):
        print(f'Invoked with args: {kwargs}')
        # 'temperature': 0.0, 'max_new_tokens': 1, 'top_p': 1.0, 'pad_token_id': 0, 'logits_processor': [<guidance.llms._transformers.BiasLogitsProcessor object at 0x7f217d161240>], 'stopping_criteria': [<guidance.llms._transformers.RegexStoppingCriteria object at 0x7f217d161180>], 'output_scores': True, 'return_dict_in_generate': True}
        generator = ExLlamaGenerator(self.exllama_model, self.exllama_tokenizer, self.exllama_cache)
        generator.settings.temperature = kwargs['temperature']
        generator.settings.top_p = kwargs['top_p']
        generator.settings.top_k = 40
        generator.settings.temperature = 0.7
        generator.settings.token_repetition_penalty_max = 1.13

        generator.end_beam_search()
        test_ids = generator.tokenizer.encode('just some stupid test')
        print(f'test_ids: {test_ids}')
        generator.gen_begin(kwargs['inputs'])
        tok = generator.gen_single_token()

        print(f'generated single token: {tok}')

        return tok

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ):
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

class ExLlamaTokenizerWrapper:
    vocab_size = 32000

    def __init__(self, tokenizer: ExLlamaTokenizer):
        self.tokenizer = tokenizer


# Config found from gptq:
# config: LlamaConfig {
#   "_name_or_path": "models/tulu-13B-GPTQ",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 5120,
#   "initializer_range": 0.02,
#   "intermediate_size": 13824,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 40,
#   "num_hidden_layers": 40,
#   "pad_token_id": 0,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float32",
#   "transformers_version": "4.30.2",
#   "use_cache": true,
#   "vocab_size": 32001
# }

# config = {
#   "_name_or_path": "models/tulu-13B-GPTQ",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 5120,
#   "initializer_range": 0.02,
#   "intermediate_size": 13824,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 40,
#   "num_hidden_layers": 40,
#   "pad_token_id": 0,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": False,
#   "torch_dtype": "float32",
#   "transformers_version": "4.30.2",
#   "use_cache": True,
#   "vocab_size": 32001
# }