from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
import torch.nn.functional as F

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)
        self.task_num = getattr(config, "task_num", 2)
        self.task_loss_weight = getattr(config, "task_loss_weight", 1.0)
        self.enable_task_prediction = getattr(config, "enable_task_prediction", True)
        self.task_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.task_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, self.task_num),
        )

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        self.post_init()
        nn.init.normal_(self.task_token, mean=0.0, std=getattr(config, "initializer_range", 0.02))

    def initialize_task_prediction_modules(self):
        dtype = next((p.dtype for p in self.language_model.parameters() if not p.is_meta), torch.float32)
        device = next((p.device for p in self.language_model.parameters() if not p.is_meta), torch.device("cpu"))
        std = getattr(self.config, "initializer_range", getattr(self.config.text_config, "initializer_range", 0.02))

        if self.task_token.is_meta:
            self.task_token = nn.Parameter(torch.empty(1, 1, self.config.hidden_size, device=device, dtype=dtype))
            nn.init.normal_(self.task_token, mean=0.0, std=std)

        for module in self.task_classifier.modules():
            for name, param in list(module._parameters.items()):
                if param is None or not param.is_meta:
                    continue
                new_param = nn.Parameter(torch.empty(param.shape, device=device, dtype=dtype), requires_grad=param.requires_grad)
                module._parameters[name] = new_param
                if isinstance(module, nn.Linear):
                    if name == "weight":
                        nn.init.normal_(new_param, mean=0.0, std=std)
                    elif name == "bias":
                        nn.init.zeros_(new_param)

    def materialize_meta_parameters(self):
        dtype = next((p.dtype for p in self.parameters() if not p.is_meta), torch.float32)
        device = next((p.device for p in self.parameters() if not p.is_meta), torch.device("cpu"))
        std = getattr(self.config, "initializer_range", getattr(self.config.text_config, "initializer_range", 0.02))
        materialized = []

        for module_name, module in self.named_modules():
            for param_name, param in list(module._parameters.items()):
                if param is None or not param.is_meta:
                    continue
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if full_name != "task_token" and "task_classifier" not in full_name:
                    continue
                new_param = nn.Parameter(
                    torch.empty(param.shape, device=device, dtype=dtype),
                    requires_grad=param.requires_grad,
                )
                module._parameters[param_name] = new_param
                if param_name == "bias":
                    nn.init.zeros_(new_param)
                elif param.ndim <= 1 or param_name == "bias":
                    nn.init.zeros_(new_param)
                else:
                    nn.init.normal_(new_param, mean=0.0, std=std)
                materialized.append(full_name)

            for buffer_name, buffer in list(module._buffers.items()):
                if buffer is None or not buffer.is_meta:
                    continue
                full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                if "task_classifier" not in full_name:
                    continue
                module._buffers[buffer_name] = torch.zeros(buffer.shape, device=device, dtype=buffer.dtype)
                materialized.append(full_name)

        return materialized

    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        task_ids: Optional[torch.LongTensor] = None,
        task_prediction_only: bool = False,
        answer_type=None,
        sample_indices=None,
        conversations=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        need_task_prediction = self.enable_task_prediction and task_ids is not None
        task_token_positions = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        lm_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if need_task_prediction or task_prediction_only:
            (
                task_inputs_embeds,
                task_attention_mask,
                task_position_ids,
                task_token_positions,
            ) = self._build_task_branch_inputs(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            task_outputs = self.language_model.forward(
                input_ids=None,
                attention_mask=task_attention_mask,
                position_ids=task_position_ids,
                past_key_values=None,
                inputs_embeds=task_inputs_embeds,
                labels=None,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
            task_logits = self._compute_task_logits(task_outputs.hidden_states[-1], task_token_positions)
            predicted_task_ids = task_logits.argmax(dim=-1)
            self._last_task_logits = task_logits
            self._last_predicted_task_ids = predicted_task_ids

            task_loss = None
            if task_ids is not None:
                task_ids = task_ids.to(device=task_logits.device, dtype=torch.long)
                task_loss = F.cross_entropy(task_logits.float(), task_ids)
                self._last_task_loss = task_loss.detach()
            else:
                self._last_task_loss = None

            if task_prediction_only:
                return CausalLMOutputWithPast(
                    loss=task_loss,
                    logits=task_logits,
                    past_key_values=task_outputs.past_key_values,
                    hidden_states=task_outputs.hidden_states,
                    attentions=task_outputs.attentions,
                )
        else:
            task_loss = None

        lm_outputs = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if need_task_prediction else return_dict
        )
        if not need_task_prediction:
            if hasattr(lm_outputs, "loss") and lm_outputs.loss is not None:
                self._last_answer_loss = lm_outputs.loss.detach()
                self._last_total_loss = lm_outputs.loss.detach()
            self._last_task_loss = None
            return lm_outputs

        total_loss = lm_outputs.loss
        if task_loss is not None:
            total_loss = task_loss * self.task_loss_weight if total_loss is None else total_loss + task_loss * self.task_loss_weight
        self._last_answer_loss = lm_outputs.loss.detach() if lm_outputs.loss is not None else None
        self._last_total_loss = total_loss.detach() if total_loss is not None else None

        if need_task_prediction:
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=lm_outputs.logits,
                past_key_values=lm_outputs.past_key_values,
                hidden_states=lm_outputs.hidden_states if output_hidden_states else None,
                attentions=lm_outputs.attentions,
            )

        if not lm_return_dict:
            outputs = lm_outputs[1:] if lm_outputs.loss is not None else lm_outputs
            return (total_loss,) + outputs

        return lm_outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            if self.enable_task_prediction:
                task_inputs_embeds, task_attention_mask, task_position_ids, task_token_positions = self._build_task_branch_inputs(
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                self._predict_and_set_task_ids(task_inputs_embeds, task_attention_mask, task_position_ids, task_token_positions)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)
            if self.enable_task_prediction:
                task_inputs_embeds, task_attention_mask, task_position_ids, task_token_positions = self._build_task_branch_inputs(
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                self._predict_and_set_task_ids(task_inputs_embeds, task_attention_mask, task_position_ids, task_token_positions)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def _build_task_branch_inputs(self, inputs_embeds, labels=None, attention_mask=None, position_ids=None):
        if inputs_embeds is None:
            return inputs_embeds, attention_mask, position_ids, None

        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        if attention_mask is None:
            attention_mask_bool = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            restore_attention_dtype = None
        else:
            restore_attention_dtype = attention_mask.dtype
            attention_mask_bool = attention_mask.bool()

        prefix_embeds = []
        for batch_idx in range(batch_size):
            valid_positions = torch.where(attention_mask_bool[batch_idx])[0]
            if valid_positions.numel() == 0:
                prefix_embeds.append(inputs_embeds[batch_idx, :0])
                continue
            else:
                valid_len = valid_positions[-1].item() + 1

            answer_start = valid_len
            if labels is not None:
                supervised_positions = torch.where(labels[batch_idx] != IGNORE_INDEX)[0]
                supervised_positions = supervised_positions[supervised_positions < valid_len]
                if supervised_positions.numel() > 0:
                    answer_start = supervised_positions[0].item()
            prefix_positions = valid_positions[valid_positions < answer_start]
            prefix_embeds.append(inputs_embeds[batch_idx, prefix_positions])

        max_len = max(x.shape[0] for x in prefix_embeds) + 1
        task_input_embeds = inputs_embeds.new_zeros((batch_size, max_len, hidden_size))
        task_attention_mask = attention_mask_bool.new_zeros((batch_size, max_len))

        task_token = self.task_token.to(device=device, dtype=inputs_embeds.dtype).expand(batch_size, -1, -1)
        task_token_positions = torch.empty(batch_size, dtype=torch.long, device=device)

        for batch_idx, cur_prefix_embeds in enumerate(prefix_embeds):
            prefix_len = cur_prefix_embeds.shape[0]
            task_input_embeds[batch_idx, :prefix_len] = cur_prefix_embeds
            task_input_embeds[batch_idx, prefix_len] = task_token[batch_idx, 0]
            task_attention_mask[batch_idx, :prefix_len + 1] = True
            task_token_positions[batch_idx] = prefix_len

        if restore_attention_dtype is not None:
            task_attention_mask = task_attention_mask.to(dtype=restore_attention_dtype)
        else:
            task_attention_mask = None

        task_position_ids = None
        if position_ids is not None:
            task_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for batch_idx in range(batch_size):
                valid_len = int(task_token_positions[batch_idx].item()) + 1
                task_position_ids[batch_idx, :valid_len] = torch.arange(valid_len, dtype=position_ids.dtype, device=position_ids.device)

        return task_input_embeds, task_attention_mask, task_position_ids, task_token_positions

    def _compute_task_logits(self, hidden_states, task_token_positions):
        if task_token_positions is None:
            task_token_positions = torch.full(
                (hidden_states.shape[0],),
                hidden_states.shape[1] - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        task_hidden = hidden_states[batch_indices, task_token_positions]
        return self.task_classifier(task_hidden)

    def _set_mora_adapters_enabled(self, enabled):
        previous_states = []
        for module in self.modules():
            if hasattr(module, 'disable_adapters') and not callable(getattr(module, 'disable_adapters')):
                previous_states.append((module, module.disable_adapters))
                module.disable_adapters = not enabled
        return previous_states

    def _restore_mora_adapter_states(self, previous_states):
        for module, previous_state in previous_states:
            module.disable_adapters = previous_state

    def _predict_and_set_task_ids(self, inputs_embeds, attention_mask, position_ids, task_token_positions):
        with torch.no_grad():
            self._set_moe_task_ids(None)
            adapter_states = self._set_mora_adapters_enabled(enabled=False)
            try:
                outputs = self.language_model.forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                self._restore_mora_adapter_states(adapter_states)
            task_logits = self._compute_task_logits(outputs.hidden_states[-1], task_token_positions)
            predicted_task_ids = task_logits.argmax(dim=-1)
            self._last_task_logits = task_logits
            self._last_predicted_task_ids = predicted_task_ids
            self._set_moe_task_ids(predicted_task_ids)

    def _set_moe_task_ids(self, task_ids):
        llm = self.language_model if hasattr(self, "language_model") else None
        if llm is not None and hasattr(llm, "model") and hasattr(llm.model, "layers"):
            for layer in llm.model.layers:
                if hasattr(layer, "mlp"):
                    layer.mlp._task_id = task_ids
        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

            

        
        
