from transformers import Phi3ForCausalLM, AutoTokenizer
# The LLM you want to add along with its corresponding tokenizer.

from . import register_llm

# Add GemmaForCausalLM along with its corresponding tokenizer and handle special tokens.
@register_llm('phi4') 
@register_llm('phi-4')
def return_gemmaclass(): 
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
        return tokenizer
    print("phi-4 registered")
    return (Phi3ForCausalLM, (AutoTokenizer, tokenizer_and_post_load))