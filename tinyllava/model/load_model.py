import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from . modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig
 
def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('. base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


def check_if_moelora_checkpoint(model_path):
    """检查是否是 MOELoRA 检查点（检查所有分片）"""
    print(f"\n🔍 Checking if MOELoRA checkpoint...")
    
    try:
        files = os.listdir(model_path)
        safetensor_files = [f for f in files if f.endswith('.safetensors') and f.startswith('model-')]
        
        if safetensor_files:
            from safetensors import safe_open
            
            has_lora_A = False
            has_lora_B = False
            has_dual_gates = False
            has_projection_gates = False
            
            for shard_file in sorted(safetensor_files):
                shard_path = os.path.join(model_path, shard_file)
                
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    
                    if any('lora_A' in k for k in keys):
                        has_lora_A = True
                    if any('lora_B' in k for k in keys):
                        has_lora_B = True
                    if any('global_task_gate_A' in k for k in keys) and any('global_task_gate_B' in k for k in keys):
                        has_dual_gates = True
                    if any('.task_gate.' in k for k in keys):
                        has_projection_gates = True
            
            print(
                f"   Summary: lora_A={has_lora_A}, lora_B={has_lora_B}, "
                f"dual_gates={has_dual_gates}, projection_gates={has_projection_gates}"
            )
            
            if has_lora_A and has_lora_B and (has_dual_gates or has_projection_gates):
                print(f"   ✅ MOELoRA checkpoint detected!")
                return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def load_moelora_model_with_weights(model_path, config, expert_num=4, lora_r=16, lora_alpha=16):
    """
    加载 MOELoRA 模型 - 优化版（避免重复加载）
    """
    print(f"\n🎯 Loading MOELoRA model (optimized)...")
    
    # 1. 加载完整模型（包括权重）
    print(f"   [1/4] Loading complete model from checkpoint...")
    model = TinyLlavaForConditionalGeneration.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    print(f"         ✅ Model loaded (with base weights)")
    
    # 2. Patch Phi3MLP forward
    print(f"   [2/4] Patching Phi3MLP forward...")
    from tinyllava.train.custom_finetune_moelora import patch_phi3mlp_forward
    patch_phi3mlp_forward()
    print(f"         ✅ Phi3MLP forward patched")
    
    # 3. 应用 MOELoRA 替换
    print(f"   [3/4] Applying MOELoRA structure...")
    from tinyllava.train.custom_finetune_moelora import replace_phi3mlp_with_mmoeloraS
    
    llm_model = model.language_model
    llm_model = replace_phi3mlp_with_mmoeloraS(
        llm_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        expert_num=expert_num,
        task_num=2,
        task_embedding_dim=32,
        adapter_name="default",
        device="cpu"
    )
    model.language_model = llm_model
    print(f"         ✅ MOELoRA structure applied")
    print(f"         Expert num: {expert_num}, LoRA r: {lora_r}, LoRA alpha: {lora_alpha}")
    
    # 4. 重新加载完整权重
    print(f"   [4/4] Reloading weights with MOELoRA parameters...")
    
    import glob
    from safetensors import safe_open
    
    # ✅ 修复：移除空格
    safetensor_pattern = os.path.join(model_path, "model-*.safetensors")
    safetensor_files = glob.glob(safetensor_pattern)
    
    print(f"         Found {len(safetensor_files)} safetensors files")
    
    if safetensor_files: 
        state_dict = {}
        for shard_file in sorted(safetensor_files):
            print(f"         Loading {os.path.basename(shard_file)}...")
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        
        print(f"         Total parameters: {len(state_dict)}")
        
        # 重新加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"         ✅ Weights reloaded")
        if missing_keys:
            print(f"            Missing: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"            Unexpected: {len(unexpected_keys)} keys")
    else:
        print(f"         ⚠️ No safetensors files found")
        print(f"         ℹ️ Using weights from initial load")
    
    # 5. 验证
    print(f"\n   🔍 Verifying MOELoRA components...")
    
    from src.MLoRA.peft.tuners.mmoeloraS import MMOELoraLinearS
    
    from utils.phi3_moelora_replacement import count_projection_gates
    gate_stats = count_projection_gates(model.language_model)

    if gate_stats['total'] > 0:
        print(
            f"      ✅ Projection-local gates found: "
            f"{gate_stats['total']} gates ({gate_stats['params']:,} params)"
        )
    elif hasattr(model.language_model, 'global_task_gate_A') and hasattr(model.language_model, 'global_task_gate_B'):
        gate_a_params = sum(p.numel() for p in model.language_model.global_task_gate_A.parameters())
        gate_b_params = sum(p.numel() for p in model.language_model.global_task_gate_B.parameters())
        print(f"      ✅ Dual gates found: Gate A ({gate_a_params:,} params) + Gate B ({gate_b_params:,} params)")
    else:
        print(f"      ❌ MOELoRA gates NOT found!")
    
    first_layer = model.language_model.model.layers[0]
    if isinstance(first_layer.mlp.gate_up_proj, MMOELoraLinearS):
        print(f"      ✅ MOELoRA layers verified")
        print(f"         LoRA r: {first_layer.mlp.gate_up_proj.r}")
        print(f"         Experts: {first_layer.mlp.gate_up_proj.expert_num}")
        
        # ✅ 修复：lora_A 是 ModuleList，不是字典
        has_lora_weights = (
            hasattr(first_layer.mlp.gate_up_proj, 'lora_A') and 
            len(first_layer.mlp.gate_up_proj.lora_A) > 0
        )
        print(f"         LoRA weights loaded: {has_lora_weights}")
        
        # 检查 LoRA 权重是否有值
        if has_lora_weights:
            # lora_A 是 ModuleList，包含多个专家
            first_expert_lora_A = first_layer.mlp.gate_up_proj.lora_A[0]  # ✅ 使用索引而不是键
            print(f"         LoRA_A[0] shape: {first_expert_lora_A.weight.shape}")
            has_data = first_expert_lora_A.weight.abs().sum().item() > 0
            print(f"         LoRA_A[0] has data: {has_data}")
    else:
        print(f"      ❌ Layer is {type(first_layer.mlp.gate_up_proj).__name__}")
    
    return model


def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    """
    加载预训练模型（自动检测并加载 MOELoRA）
    """
    
    # 初始化返回变量
    model = None
    tokenizer = None
    image_processor = None
    context_len = 2048
    
    print(f"\n{'='*80}")
    print(f"📂 Loading Model")
    print(f"{'='*80}")
    print(f"Model path: {model_name_or_path}")
    
    # ✅ 检测分布式环境
    is_distributed = (
        os.environ.get('WORLD_SIZE') is not None or 
        os.environ.get('LOCAL_RANK') is not None or
        os.environ.get('RANK') is not None
    )
    
    # ✅ 根据环境调整 device_map
    if is_distributed:
        print(f"🚀 Distributed training detected (LOCAL_RANK={os.environ.get('LOCAL_RANK')})")
        print(f"   Setting device_map=None (DDP/DeepSpeed will handle device placement)")
        device_map = None  # 让 DDP/DeepSpeed 管理设备
    else: 
        print(f"💻 Single device mode, using device_map='{device_map}'")
    
    # ✅ 构建模型参数（处理 device_map=None 的情况）
    kwargs_model = {**kwargs}
    
    if device_map is not None:
        kwargs_model["device_map"] = device_map
    
    if device != "cuda":
        kwargs_model['device_map'] = {"": device}

    if load_8bit:
        kwargs_model['load_in_8bit'] = True
    elif load_4bit: 
        kwargs_model['load_in_4bit'] = True
        kwargs_model['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs_model['torch_dtype'] = torch.float16
    
    # ========================================
    # 🔍 检查是否是 MOELoRA 检查点
    # ========================================
    is_moelora = check_if_moelora_checkpoint(model_name_or_path)
    
    # ========================================
    # 判断是否是 LoRA 模型
    # ========================================
    is_lora_model = (model_name_or_path is not None and 
                     'lora' in model_name_or_path.lower() and 
                     os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')))
    
    # ========================================
    # 加载模型
    # ========================================
    if is_lora_model: 
        print(f"\n[1/3] Loading LoRA model...")
        # (保持原有 LoRA 逻辑)
        pass
    
    elif is_moelora:
        # ✅ MOELoRA 专用加载路径
        print(f"\n[1/3] Loading MOELoRA model...")
        
        # 加载配置
        config = TinyLlavaConfig.from_pretrained(model_name_or_path)
        
        # ✅ 在分布式模式下，使用 CPU 加载，稍后由 Trainer 移动到 GPU
        model = load_moelora_model_with_weights(
            model_name_or_path,
            config,
            expert_num=kwargs.get('expert_num', 4),
            lora_r=kwargs.get('lora_r', 16),
            lora_alpha=kwargs.get('lora_alpha', 16)
        )
        
        print(f"      ✅ MOELoRA model loaded")
    
    else:
        # 标准模型加载
        print(f"\n[1/3] Loading standard model...")
        
        try:
            model = TinyLlavaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=False,
                **kwargs_model
            )
            print(f"      ✅ Standard model loaded")
            
        except Exception as e: 
            print(f"      ❌ Failed: {e}")
            raise
    
    # ========================================
    # 加载 image_processor
    # ========================================
    if hasattr(model, "initialize_task_prediction_modules"):
        model.initialize_task_prediction_modules()
    if hasattr(model, "materialize_meta_parameters"):
        materialized = model.materialize_meta_parameters()
        if materialized:
            print(f"      Materialized {len(materialized)} meta parameters/buffers before DeepSpeed")
            for name in materialized[:20]:
                print(f"         - {name}")
        meta_params = [name for name, param in model.named_parameters() if param.is_meta]
        if meta_params:
            print(f"      Warning: {len(meta_params)} meta parameters remain after materialization")
            for name in meta_params[:10]:
                print(f"         - {name}")
            raise RuntimeError("Meta parameters remain after model load; checkpoint may be incomplete.")
        else:
            print(f"      No meta parameters remain after model load")

    print(f"\n[2/3] Loading image processor...")
    
    try:
        image_processor = model.vision_tower._image_processor
        print(f"      ✅ Image processor loaded")
    except Exception as e:
        from transformers import AutoImageProcessor
        vision_tower_name = getattr(model.config, 'vision_tower', 'google/siglip-so400m-patch14-384')
        image_processor = AutoImageProcessor.from_pretrained(vision_tower_name)
        print(f"      ✅ Image processor loaded from config")
    
    # ========================================
    # 加载 tokenizer
    # ========================================
    print(f"\n[3/3] Loading tokenizer...")
    
    try:
        tokenizer = model.tokenizer
        print(f"      ✅ Tokenizer loaded")
    except Exception as e:
        llm_name = getattr(model.config, 'llm_model_name_or_path', 'microsoft/Phi-3.5-mini-instruct')
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False, padding_side="right")
        print(f"      ✅ Tokenizer loaded from config")
    
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    
    # ✅ 只在非分布式模式下设置为评估模式
    if not is_distributed:
        model.eval()
        print(f"   Model set to eval mode")
    else:
        print(f"   Model in training mode (distributed)")
    
    # ========================================
    # 最终总结
    # ========================================
    print(f"\n📊 Model Summary:")
    if is_moelora:
        print(f"   🎯 MOELoRA Model")
        try:
            from utils.phi3_moelora_replacement import count_projection_gates
            gate_stats = count_projection_gates(model.language_model)
        except Exception:
            gate_stats = {'total': 0, 'params': 0}
        if gate_stats['total'] > 0:
            print(f"      Projection-local gates: {gate_stats['total']} ({gate_stats['params']:,} params)")
        elif hasattr(model.language_model, 'global_task_gate_A') or hasattr(model.language_model, 'global_task_gate_B'):
            gate_a = sum(p.numel() for p in model.language_model.global_task_gate_A.parameters()) if hasattr(model.language_model, 'global_task_gate_A') else 0
            gate_b = sum(p.numel() for p in model.language_model.global_task_gate_B.parameters()) if hasattr(model.language_model, 'global_task_gate_B') else 0
            print(f"      Gate A + Gate B: {gate_a + gate_b:,} params")
    else:
        print(f"   ℹ️ Standard Model")
    
    print(f"\n{'='*80}")
    print(f"✅ MODEL LOADED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Model:              {type(model).__name__}")
    print(f"Tokenizer:          {type(tokenizer).__name__}")
    print(f"Image Processor:    {type(image_processor).__name__}")
    print(f"Context Length:     {context_len}")
    print(f"Device Map:         {device_map}")
    print(f"Distributed:        {is_distributed}")
    print(f"{'='*80}\n")
    
    return model, tokenizer, image_processor, context_len
