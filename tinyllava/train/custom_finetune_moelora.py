"""
带 MMOELoRA 支持的自定义微调脚本（完整版）
支持 answer_type → task_id → 专家选择 + 双门控
支持每100个step打印门控权重
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import tokenizers
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

# TinyLLaVA 组件
from tinyllava.train.tinyllava_trainer_moelora import MOELoRATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils.arguments import *
from tinyllava.utils.logging import log_trainable_params
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.data import *

# MOELoRA 工具
from utils.phi3_moelora_replacement import (
    replace_phi3mlp_with_mmoeloraS,
    patch_phi3mlp_forward,
    verify_replacement,
    verify_gate_sharing,
    print_parameter_stats
)
from src.MLoRA.peft.tuners.mmoeloraS import MMOELoraLinearS


# ✅ 自定义回调函数：定期打印门控权重
class DualGateWeightCallback(transformers.TrainerCallback):
    """
    每100个step打印一次双门控权重分布
    """
    def __init__(self, llm_model, print_interval=100):
        self.llm_model = llm_model
        self.print_interval = print_interval
        self.last_print_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """每步结束后检查是否需要打印"""
        current_step = state.global_step
        
        # 每 print_interval 个 step 打印一次（默认 100）
        if current_step > 0 and (current_step - self.last_print_step) >= self.print_interval:
            self._print_gate_weights(state)
            self.last_print_step = current_step
    
    def _print_gate_weights(self, state):
        """打印门控权重"""
        print("\n" + "="*80)
        print(f"🎯 EXPERT GATE WEIGHTS (Step {state.global_step}, Epoch {state.epoch:.2f})")
        print("="*80)
        
        # 任务映射
        task_mapping = {
            'OPEN': 0,
            'CLOSED': 1
        }
        
        if not (hasattr(self.llm_model, 'global_task_gate_A') and hasattr(self.llm_model, 'global_task_gate_B')):
            print("⚠️ Dual gates not found!")
            return
        
        with torch.no_grad():
            for task_name, task_id in task_mapping.items():
                # 获取门控设备
                gate_a_device = next(self.llm_model.global_task_gate_A.parameters()).device
                gate_b_device = next(self.llm_model.global_task_gate_B.parameters()).device
                
                task_tensor_a = torch.tensor([task_id], dtype=torch.long, device=gate_a_device)
                task_tensor_b = torch.tensor([task_id], dtype=torch.long, device=gate_b_device)
                
                print(f"\n📊 {task_name} (Task ID={task_id}):")
                
                # Gate A (gate_up_proj)
                expert_weights_a = self.llm_model.global_task_gate_A(task_tensor_a)
                print(f"  Gate A (gate_up_proj):")
                print(f"    Expert weights: {expert_weights_a[0].cpu().numpy()}")
                
                max_expert_a = expert_weights_a[0].argmax().item()
                for i, weight in enumerate(expert_weights_a[0].cpu().numpy()):
                    bar_length = int(weight * 40)
                    bar = "█" * bar_length
                    marker = " ⭐" if i == max_expert_a else ""
                    print(f"      Expert {i}: {weight:.4f} |{bar:<40}|{marker}")
                
                # Gate B (down_proj)
                expert_weights_b = self.llm_model.global_task_gate_B(task_tensor_b)
                print(f"  Gate B (down_proj):")
                print(f"    Expert weights: {expert_weights_b[0].cpu().numpy()}")
                
                max_expert_b = expert_weights_b[0].argmax().item()
                for i, weight in enumerate(expert_weights_b[0].cpu().numpy()):
                    bar_length = int(weight * 40)
                    bar = "█" * bar_length
                    marker = " ⭐" if i == max_expert_b else ""
                    print(f"      Expert {i}: {weight:.4f} |{bar:<40}|{marker}")
                
                # 计算熵值
                entropy_a = -torch.sum(expert_weights_a[0] * torch.log(expert_weights_a[0] + 1e-10)).item()
                entropy_b = -torch.sum(expert_weights_b[0] * torch.log(expert_weights_b[0] + 1e-10)).item()
                
                print(f"  Entropy: Gate A={entropy_a:.4f}, Gate B={entropy_b:.4f}")
        
        print("="*80 + "\n")


def load_settings(model_arguments, data_arguments, training_arguments):
    """加载设置"""
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio


def analyze_model_parameters(model):
    """
    详细分析模型参数梯度状态
    """
    print("\n" + "="*80)
    print("🔍 DETAILED PARAMETER ANALYSIS")
    print("="*80)
    
    # 统计分类
    stats = {
        'total': 0,
        'trainable': 0,
        'frozen': 0,
        'lora_A': 0,
        'lora_B': 0,
        'global_gate': 0,
        'connector': 0,
        'vision_tower': 0,
        'language_model': 0,
        'other': 0
    }
    
    # 按模块分组
    module_stats = {}
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        stats['total'] += num_params
        
        # 获取模块名称
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in module_stats:
            module_stats[module_name] = {'trainable': 0, 'frozen': 0}
        
        if param.requires_grad:
            stats['trainable'] += num_params
            module_stats[module_name]['trainable'] += num_params
            trainable_params.append((name, num_params, param.dtype, param.device))
            
            # 分类统计 - ✅ 支持双门控
            if 'lora_A' in name:
                stats['lora_A'] += num_params
            elif 'lora_B' in name:
                stats['lora_B'] += num_params
            elif 'global_task_gate_A' in name or 'global_task_gate_B' in name:
                stats['global_gate'] += num_params
            elif 'connector' in name:
                stats['connector'] += num_params
            elif 'vision_tower' in name:
                stats['vision_tower'] += num_params
            elif 'language_model' in name:
                stats['language_model'] += num_params
            else:
                stats['other'] += num_params
        else: 
            stats['frozen'] += num_params
            module_stats[module_name]['frozen'] += num_params
            frozen_params.append((name, num_params, param.dtype, param.device))
    
    # 打印总览
    print(f"\n📊 PARAMETER OVERVIEW")
    print(f"{'='*80}")
    print(f"{'Category':<30} {'Count':>15} {'Percentage':>15}")
    print(f"{'-'*80}")
    print(f"{'Total Parameters':<30} {stats['total']:>15,} {100.0:>14.2f}%")
    print(f"{'├─ Trainable (requires_grad=True)':<30} {stats['trainable']:>15,} {100*stats['trainable']/stats['total']:>14.2f}%")
    print(f"{'└─ Frozen (requires_grad=False)':<30} {stats['frozen']:>15,} {100*stats['frozen']/stats['total']:>14.2f}%")
    
    # 可训练参数细分
    if stats['trainable'] > 0:
        print(f"\n📋 TRAINABLE PARAMETER BREAKDOWN")
        print(f"{'='*80}")
        print(f"{'Type':<30} {'Count':>15} {'% of Trainable':>15}")
        print(f"{'-'*80}")
        
        categories = [
            ('LoRA A Matrices', stats['lora_A']),
            ('LoRA B Matrices', stats['lora_B']),
            ('Global Task Gates (A + B)', stats['global_gate']),
            ('Connector', stats['connector']),
            ('Vision Tower', stats['vision_tower']),
            ('Language Model (other)', stats['language_model']),
            ('Other', stats['other'])
        ]
        
        for cat_name, cat_count in categories: 
            if cat_count > 0:
                percentage = 100 * cat_count / stats['trainable']
                print(f"{cat_name:<30} {cat_count:>15,} {percentage:>14.2f}%")
    
    # 按模块统计
    print(f"\n🏗️ MODULE-WISE BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Module':<40} {'Trainable':>18} {'Frozen':>18}")
    print(f"{'-'*80}")
    
    for module_name in sorted(module_stats.keys()):
        trainable = module_stats[module_name]['trainable']
        frozen = module_stats[module_name]['frozen']
        total_module = trainable + frozen
        
        if total_module > 0:
            trainable_pct = 100 * trainable / total_module if total_module > 0 else 0
            print(f"{module_name:<40} {trainable:>12,} ({trainable_pct:>4.1f}%) {frozen:>12,}")
    
    # 显示前20个可训练参数
    print(f"\n✅ TOP 20 TRAINABLE PARAMETERS")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Parameter Name':<50} {'Shape':>15} {'Device':>8}")
    print(f"{'-'*80}")
    
    trainable_sorted = sorted(trainable_params, key=lambda x: x[1], reverse=True)
    for i, (name, num_params, dtype, device) in enumerate(trainable_sorted[:20], 1):
        try:
            param = dict(model.named_parameters())[name]
            shape_str = str(tuple(param.shape))
        except:
            shape_str = f"{num_params:,}"
        
        device_str = str(device).split(':')[0]
        print(f"{i:<4} {name[:48]:<50} {shape_str:>15} {device_str:>8}")
    
    if len(trainable_sorted) > 20:
        remaining = len(trainable_sorted) - 20
        remaining_params = sum(p[1] for p in trainable_sorted[20:])
        print(f"{'.. .':<4} {'...  and ' + str(remaining) + ' more parameters':<50} {remaining_params:>15,}")
    
    # 显示前10个冻结参数
    print(f"\n❄️ SAMPLE FROZEN PARAMETERS (First 10)")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Parameter Name':<50} {'Shape':>15} {'Device':>8}")
    print(f"{'-'*80}")
    
    for i, (name, num_params, dtype, device) in enumerate(frozen_params[:10], 1):
        try:
            param = dict(model.named_parameters())[name]
            shape_str = str(tuple(param.shape))
        except:
            shape_str = f"{num_params:,}"
        
        device_str = str(device).split(':')[0]
        print(f"{i:<4} {name[:48]:<50} {shape_str:>15} {device_str:>8}")
    
    if len(frozen_params) > 10:
        remaining = len(frozen_params) - 10
        remaining_params = sum(p[1] for p in frozen_params[10:])
        print(f"{'...':<4} {'... and ' + str(remaining) + ' more frozen parameters':<50} {remaining_params:>15,}")
    
    # 验证
    print(f"\n✅ VERIFICATION")
    print(f"{'='*80}")
    computed_total = stats['trainable'] + stats['frozen']
    match_status = "✅ MATCH" if computed_total == stats['total'] else "⚠️ MISMATCH"
    print(f"Trainable + Frozen = {computed_total:,}")
    print(f"Total Parameters   = {stats['total']:,}")
    print(f"Status: {match_status}")
    
    if stats['trainable'] == 0:
        print(f"\n⚠️ WARNING: No trainable parameters found!  Training will not work.")
    
    print(f"{'='*80}\n")
    
    return stats

def create_optimizer_with_custom_lr(model, llm_model, training_arguments):
    """
    创建带有差分学习率的优化器
    - Gate 参数: learning_rate × 10
    - 其他参数: learning_rate × 1
    """
    base_lr = training_arguments.learning_rate
    gate_lr = 0.005  # Gate 学习率为 10 倍
    
    print("\n" + "="*80)
    print("🎯 CREATING OPTIMIZER WITH DIFFERENTIAL LEARNING RATES")
    print("="*80)
    print(f"Base learning rate:        {base_lr}")
    print(f"Gate learning rate:       {gate_lr} (10x)")
    print("="*80 + "\n")
    
    # 分离参数组
    gate_params = []
    other_params = []
    
    gate_param_names = []
    other_param_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # ✅ 检查双门控
            if 'global_task_gate_A' in name or 'global_task_gate_B' in name:
                gate_params.append(param)
                gate_param_names.append(name)
            else:
                other_params.append(param)
                other_param_names.append(name)
    
    # 统计信息
    gate_count = sum(p.numel() for p in gate_params)
    other_count = sum(p.numel() for p in other_params)
    total_trainable = gate_count + other_count
    
    print(f"📊 PARAMETER GROUP STATISTICS:")
    print(f"{'='*80}")
    print(f"{'Group':<30} {'Parameters':>15} {'Learning Rate':>20}")
    print(f"{'-'*80}")
    print(f"{'Gates (global_task_gate_A/B)':<30} {gate_count:>15,} {gate_lr:>20.2e}")
    print(f"{'Other (LoRA, etc.)':<30} {other_count:>15,} {base_lr:>20.2e}")
    print(f"{'-'*80}")
    print(f"{'Total Trainable':<30} {total_trainable:>15,}")
    print(f"{'='*80}\n")
    
    # 打印 Gate 参数详情
    if gate_param_names:
        print(f"✅ Gate Parameters (LR = {gate_lr:.2e}):")
        for name in gate_param_names:
            param = dict(model.named_parameters())[name]
            print(f"   - {name:<60} {tuple(param.shape)}")
    else:
        print(f"⚠️ WARNING: No gate parameters found!")
    
    # 打印部分其他参数
    print(f"\n✅ Other Trainable Parameters (LR = {base_lr:.2e}) - showing first 10:")
    for i, name in enumerate(other_param_names[:10]):
        param = dict(model.named_parameters())[name]
        print(f"   - {name:<60} {tuple(param.shape)}")
    if len(other_param_names) > 10:
        print(f"   ... and {len(other_param_names) - 10} more parameters")
    
    print("="*80 + "\n")
    
    # 创建参数组
    param_groups = []
    
    if gate_params:
        param_groups.append({
            'params': gate_params,
            'lr': gate_lr,
            'weight_decay': training_arguments.weight_decay,
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': training_arguments.weight_decay,
        })
    
    if not param_groups:
        raise ValueError("No trainable parameters found!  Cannot create optimizer.")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_arguments.adam_beta1, training_arguments.adam_beta2),
        eps=training_arguments.adam_epsilon,
    )
    
    print(f"✅ Created AdamW optimizer with {len(param_groups)} parameter group(s)")
    print(f"   - Gate group: {len(gate_params)} tensors, LR={gate_lr:.2e}")
    print(f"   - Other group: {len(other_params)} tensors, LR={base_lr:.2e}\n")
    
    return optimizer

def train():
    # ========================================
    # 1. 解析参数
    # ========================================
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    # 简化日志设置
    output_dir = getattr(training_arguments, 'output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    load_settings(model_arguments, data_arguments, training_arguments)
    
    # ========================================
    # 2. Patch Phi3MLP forward
    # ========================================
    print("\n" + "="*60)
    print("🚀 MOELoRA Training with Dual Independent Gates")
    print("="*60)
    patch_phi3mlp_forward()
    
    # ========================================
    # 3. 加载预训练模型
    # ========================================
    print("\n📥 Loading pretrained model...")
    model, tokenizer, image_processor, context_len = load_pretrained_model(
        training_arguments.pretrained_model_path
    )
    config = model.config
    
    # ========================================
    # 4. 应用 MMOELoRA 替换
    # ========================================
    print("\n🔧 Applying MMOELoRA to LLM...")
    # ✅ 修复：使用 CPU，让 Trainer 处理设备分配
    device = "cpu"
    
    moelora_config = {
        'lora_r': getattr(training_arguments, 'lora_r', 16),
        'lora_alpha': getattr(training_arguments, 'lora_alpha', 32),
        'lora_dropout': getattr(training_arguments, 'lora_dropout', 0.1),
        'expert_num': getattr(training_arguments, 'expert_num', 4),
        'task_num': getattr(training_arguments, 'task_num', 2),
        'task_embedding_dim': getattr(training_arguments, 'task_embedding_dim', 32)
    }
    
    print(f"\n📋 MOELoRA Config:")
    for k, v in moelora_config.items():
        print(f"  {k}: {v}")
    print(f"\n📋 Task Mapping:")
    print(f"  OPEN → Task ID 0")
    print(f"  CLOSED → Task ID 1")
    print(f"\n📋 Dual Gate Configuration:")
    print(f"  Gate A: Used by gate_up_proj")
    print(f"  Gate B: Used by down_proj")
    print(f"  Both gates trained independently")
    
    # 找到并替换 LLM
    llm_model = None
    if hasattr(model, 'language_model'):
        llm_model = model.language_model
        print(f"\n📦 Found LLM: model.language_model")
    elif hasattr(model, 'llm'):
        llm_model = model.llm
        print(f"\n📦 Found LLM: model.llm")
    elif 'Phi3' in type(model).__name__:
        llm_model = model
        print(f"\n📦 Using model directly as LLM")
    else:
        raise ValueError("Cannot find LLM in model structure")
    
    print(f"   LLM type: {type(llm_model).__name__}")
    
    # 应用 MMOELoRA 替换
    llm_model = replace_phi3mlp_with_mmoeloraS(
        llm_model,
        **moelora_config,
        device=device
    )
    
    # 更新模型中的 LLM
    if hasattr(model, 'language_model'):
        model.language_model = llm_model
    elif hasattr(model, 'llm'):
        model.llm = llm_model
    
    # 验证替换
    verify_replacement(llm_model)
    verify_gate_sharing(llm_model)
    
    # ========================================
    # 5. 准备数据
    # ========================================
    data_arguments.image_processor = image_processor
    data_arguments.is_multimodal = True
    
    print("\n📦 Loading data with task_id support...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_arguments
    )
    
    # ========================================
    # 6. 应用 training_recipe
    # ========================================
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    # ========================================
    # 7. 在 training_recipe 之后重新解冻参数（关键！）
    # ========================================
    print("\n❄️ Re-freezing and unfreezing parameters after training_recipe...")
    
    # 先全部冻结
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻 LoRA 参数
    lora_count = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name: 
            param.requires_grad = True
            lora_count += 1
    print(f"  ✅ Unfrozen {lora_count} LoRA parameters")
    
    # ✅ 解冻双门控
    if hasattr(llm_model, 'global_task_gate_A') and hasattr(llm_model, 'global_task_gate_B'):
        gate_a_params = sum(p.numel() for p in llm_model.global_task_gate_A.parameters())
        gate_b_params = sum(p.numel() for p in llm_model.global_task_gate_B.parameters())
        # 解冻两个门控
        for param in llm_model.global_task_gate_A.parameters():
            param.requires_grad = True
        for param in llm_model.global_task_gate_B.parameters():
            param.requires_grad = True
        print(f"  ✅ Unfrozen dual gates: Gate A ({gate_a_params:,}) + Gate B ({gate_b_params:,})")
    
    # 解冻 connector（如果需要）
    if hasattr(model, 'connector') and training_arguments.tune_type_connector == 'full':
        for param in model.connector.parameters():
            param.requires_grad = True
        print(f"  ✅ Unfrozen connector")
    
    # 解冻 vision_tower（如果需要）
    if hasattr(model, 'vision_tower') and training_arguments.tune_type_vision_tower != 'frozen':
        for param in model.vision_tower.parameters():
            param.requires_grad = True
        print(f"  ✅ Unfrozen vision tower")
    
    # 打印参数统计
    print_parameter_stats(llm_model)
    
    # ✅ 添加详细参数分析
    param_stats = analyze_model_parameters(model)
    
    if param_stats['trainable'] == 0:
        raise ValueError("❌ No trainable parameters!  Training cannot proceed.")
    
    # ========================================
    # ✅ 新增：训练前双门控权重可视化
    # ========================================
    print("\n" + "="*80)
    print("🎯 INITIAL EXPERT GATE WEIGHTS (初始 - 双门控)")
    print("="*80)
    
    # 获取 LLM 模型
    llm_model = None
    if hasattr(model, 'language_model'):
        llm_model = model.language_model
    elif hasattr(model, 'llm'):
        llm_model = model.llm
    
    if llm_model and (hasattr(llm_model, 'global_task_gate_A') or hasattr(llm_model, 'global_task_gate_B')):
        # 任务映射
        task_mapping = {
            'OPEN': 0,
            'CLOSED': 1
        }
        
        with torch.no_grad():
            for task_name, task_id in task_mapping.items():
                # 创建 task tensor
                gate_a_device = next(llm_model.global_task_gate_A.parameters()).device if hasattr(llm_model, 'global_task_gate_A') else 'cpu'
                task_tensor = torch.tensor([task_id], dtype=torch.long, device=gate_a_device)
                
                print(f"\n📊 {task_name} (Task ID={task_id}):")
                
                # Gate A
                if hasattr(llm_model, 'global_task_gate_A'):
                    expert_weights_a = llm_model.global_task_gate_A(task_tensor)
                    print(f"  Gate A (gate_up_proj):")
                    print(f"    Expert weights: {expert_weights_a[0].cpu().numpy()}")
                    
                    max_expert_a = expert_weights_a[0].argmax().item()
                    for i, weight in enumerate(expert_weights_a[0].cpu().numpy()):
                        bar = "█" * int(weight * 40)
                        marker = " ⭐" if i == max_expert_a else ""
                        print(f"      Expert {i}: {weight:.4f} |{bar}|{marker}")
                
                # Gate B
                if hasattr(llm_model, 'global_task_gate_B'):
                    gate_b_device = next(llm_model.global_task_gate_B.parameters()).device
                    task_tensor_b = torch.tensor([task_id], dtype=torch.long, device=gate_b_device)
                    expert_weights_b = llm_model.global_task_gate_B(task_tensor_b)
                    print(f"  Gate B (down_proj):")
                    print(f"    Expert weights: {expert_weights_b[0].cpu().numpy()}")
                    
                    max_expert_b = expert_weights_b[0].argmax().item()
                    for i, weight in enumerate(expert_weights_b[0].cpu().numpy()):
                        bar = "█" * int(weight * 40)
                        marker = " ⭐" if i == max_expert_b else ""
                        print(f"      Expert {i}: {weight:.4f} |{bar}|{marker}")
    
    else:
        print("⚠️ Warning: No dual gates found in model!")
    
    print("="*80 + "\n")

    # ========================================
    # 8. 创建自定义优化器（Gate 使用 10 倍学习率）
    # ========================================
    print("\n🔧 Creating custom optimizer with differential learning rates...")
    custom_optimizer = create_optimizer_with_custom_lr(model, llm_model, training_arguments)
    
    # ========================================
    # 9. ✅ 创建双门控权重回调（每100个step打印）
    # ========================================
    gate_weight_callback = DualGateWeightCallback(llm_model, print_interval=100)
    
    # ========================================
    # 10. 使用 MOELoRATrainer（传入自定义优化器和回调）
    # ========================================
    print("\n🏋️ Initializing MOELoRATrainer with gate weight monitoring...")
    trainer = MOELoRATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=[gate_weight_callback],  # ✅ 添加回调
        optimizers=(custom_optimizer, None),  # 传入自定义优化器，scheduler 为 None（使用默认）
        **data_module
    )
    
    # ========================================
    # 11. 开始训练
    # ========================================
    print("\n🏃 Starting training...")
    print("="*60)
    print(f"📊 Gate weights will be printed every 100 steps")
    print("="*60)
    trainer.train()
    
    # ========================================
    # 12. 保存模型
    # ========================================
    print("\n💾 Saving model...")
    training_recipe.save(model, trainer)
    
    # ========================================
    # 13. 最终门控权重
    # ========================================
    print("\n" + "="*80)
    print("🎯 FINAL EXPERT GATE WEIGHTS (训练完成)")
    print("="*80)
    
    task_mapping = {
        'OPEN': 0,
        'CLOSED': 1
    }
    
    if llm_model and (hasattr(llm_model, 'global_task_gate_A') or hasattr(llm_model, 'global_task_gate_B')):
        with torch.no_grad():
            for task_name, task_id in task_mapping.items():
                gate_a_device = next(llm_model.global_task_gate_A.parameters()).device if hasattr(llm_model, 'global_task_gate_A') else 'cpu'
                task_tensor = torch.tensor([task_id], dtype=torch.long, device=gate_a_device)
                
                print(f"\n📊 {task_name} (Task ID={task_id}):")
                
                if hasattr(llm_model, 'global_task_gate_A'):
                    expert_weights_a = llm_model.global_task_gate_A(task_tensor)
                    print(f"  Gate A (gate_up_proj):")
                    print(f"    Expert weights: {expert_weights_a[0].cpu().numpy()}")
                    max_expert_a = expert_weights_a[0].argmax().item()
                    for i, weight in enumerate(expert_weights_a[0].cpu().numpy()):
                        bar = "█" * int(weight * 40)
                        marker = " ⭐" if i == max_expert_a else ""
                        print(f"      Expert {i}: {weight:.4f} |{bar:<40}|{marker}")
                
                if hasattr(llm_model, 'global_task_gate_B'):
                    gate_b_device = next(llm_model.global_task_gate_B.parameters()).device
                    task_tensor_b = torch.tensor([task_id], dtype=torch.long, device=gate_b_device)
                    expert_weights_b = llm_model.global_task_gate_B(task_tensor_b)
                    print(f"  Gate B (down_proj):")
                    print(f"    Expert weights: {expert_weights_b[0].cpu().numpy()}")
                    max_expert_b = expert_weights_b[0].argmax().item()
                    for i, weight in enumerate(expert_weights_b[0].cpu().numpy()):
                        bar = "█" * int(weight * 40)
                        marker = " ⭐" if i == max_expert_b else ""
                        print(f"      Expert {i}: {weight:.4f} |{bar:<40}|{marker}")
    
    print("="*80 + "\n")
    print("\n🎉 Training completed!")


if __name__ == "__main__":
    train()
