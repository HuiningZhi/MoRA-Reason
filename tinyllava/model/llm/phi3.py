from transformers import Phi3ForCausalLM, AutoTokenizer
# The LLM you want to add along with its corresponding tokenizer.

from . import register_llm

# Add GemmaForCausalLM along with its corresponding tokenizer and handle special tokens.
@register_llm('phi-3') 
def return_gemmaclass(): 
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
        return tokenizer
    return (Phi3ForCausalLM, (AutoTokenizer, tokenizer_and_post_load))