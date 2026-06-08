import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.data import *
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)
    # load pretrained checkpoint
    # model = AutoModelForCausalLM.from_pretrained(training_arguments.pretrained_model_path, trust_remote_code=True)
    model, tokenizer, image_processor, context_len = load_pretrained_model(training_arguments.pretrained_model_path)
    config = model.config
    # tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    data_arguments.image_processor = image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
