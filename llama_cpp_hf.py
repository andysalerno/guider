import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from guidance.llms import Transformers
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_roles import Llama2ChatRole, Llama2GuanacoRole

if torch.cuda.is_available() and not torch.version.hip:
    try:
        from llama_cpp_cuda import Llama
    except:
        print("Failed to import llama_cpp_cuda, falling back on llama_cpp")
        from llama_cpp import Llama
else:
    from llama_cpp import Llama

selected_role = None


def get_role():
    return selected_role
    # return Vicuna1_3Role
    # return Llama2ChatRole
    # return Llama2GuanacoRole
    # return Llama2UncensoredChatRole


class LlamacppHF(Transformers):
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

        tokenizer = AutoTokenizer.from_pretrained("TheBloke/StableBeluga-13B-GPTQ")

        model = LlamacppHFInner.from_pretrained(model)

        # model._update_model_kwargs_for_generation = (
        #     LlamaForCausalLM._update_model_kwargs_for_generation
        # )
        # model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        return get_role().role_start(role)

    @staticmethod
    def role_end(role):
        return get_role().role_end(role)


class LlamacppHFInner(PreTrainedModel):
    def __init__(self, model):
        super().__init__(PretrainedConfig())
        self.model = model
        self.generation_config = GenerationConfig()
        self.cache = None

        self.config.vocab_size = 32000

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        # TODO: Some decoding methods (such as Contrastive Search) may not work at this time
        assert len(args) == 0, "no *args should be passed to forward"
        use_cache = kwargs.get("use_cache", True)
        labels = kwargs.get("labels", None)
        seq = kwargs["input_ids"][0].tolist()
        cache = kwargs["past_key_values"] if "past_key_values" in kwargs else None

        # Make the forward call
        seq_tensor = torch.tensor(seq)
        if labels is None:
            if self.cache is None or not torch.equal(self.cache, seq_tensor[:-1]):
                self.model.reset()
                self.model.eval(seq)
            else:
                self.model.eval([seq[-1]])

            logits = (
                torch.tensor(self.model.scores[self.model.n_tokens - 1, :])
                .view(1, 1, -1)
                .to(kwargs["input_ids"].device)
            )
        else:
            self.model.reset()
            self.model.eval(seq)
            logits = torch.tensor(self.model.eval_logits)
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(
                kwargs["input_ids"].device
            )

        self.cache = seq_tensor

        # Based on transformers/models/llama/modeling_llama.py
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            logits=logits, past_key_values=cache if use_cache else None, loss=loss
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        path = Path(pretrained_model_name_or_path)
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob("*ggml*.bin"))[0]

        print(f"llama.cpp weights detected: {model_file}\n")
        params = {
            "model_path": str(model_file),
            "n_ctx": 4096,
            "seed": 0,
            "n_threads": 8,
            # "n_batch": 1,
            "use_mmap": True,
            "use_mlock": False,
            "low_vram": False,
            "n_gpu_layers": 20,
            # "rope_freq_base": 10000 * shared.args.alpha_value ** (64 / 63.0),
            # "rope_freq_scale": 1.0 / shared.args.compress_pos_emb,
            # "n_gqa": shared.args.n_gqa or None,
            # "rms_norm_eps": shared.args.rms_norm_eps or None,
            "logits_all": True,
        }

        model = Llama(**params)
        return LlamacppHFInner(model)
