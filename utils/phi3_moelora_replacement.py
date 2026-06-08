"""
Phi3 MLP 替换为 MMOELoraLinearS
双门控 + DeepSpeed 兼容版本
"""
import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3MLP
from src.MLoRA.peft.tuners.mmoeloraS import MMOELoraLinearS


# ✅ 温度 Softmax（可选，用于控制门控锐度）
class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1.0, dim=-1):
        super().__init__()
        self.temperature = temperature
        self.dim = dim
    
    def forward(self, x):
        return nn.functional.softmax(x / self.temperature, dim=self.dim)


# ✅ 双门控注册表
_DUAL_GATE_REGISTRY = {}


def replace_phi3mlp_with_mmoeloraS(
    model,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    expert_num=4,
    task_num=2,
    task_embedding_dim=32,
    adapter_name="default",
    device="cpu",
    temperature=1.0
):
    """
    使用双门控的 MMOELoraLinearS 替换 Phi3MLP
    
    ✅ Gate A 用于 gate_up_proj
    ✅ Gate B 用于 down_proj
    ✅ 两个门控独立训练
    ✅ DeepSpeed 兼容：通过 forward 传递门控，避免共享参数问题
    ✅ 使用全局注册表：避免循环引用导致的递归错误
    """
    print("\n🔧 Replacing Phi3MLP layers with MMOELoraLinearS (Dual Independent Gates)...")
    
    # ✅ 创建 Gate A（用于 gate_up_proj）
    gate_A = nn.Sequential(
        nn.Embedding(task_num + 1, task_embedding_dim),
        nn.Linear(task_embedding_dim, expert_num, bias=False),
        TemperatureSoftmax(temperature=temperature, dim=-1)
    )
    nn.init.normal_(gate_A[0].weight, std=0.1)
    nn.init.normal_(gate_A[1].weight, std=0.5)
    gate_A = gate_A.to(device)
    
    # ✅ 创建 Gate B（用于 down_proj）
    gate_B = nn.Sequential(
        nn.Embedding(task_num + 1, task_embedding_dim),
        nn.Linear(task_embedding_dim, expert_num, bias=False),
        TemperatureSoftmax(temperature=temperature, dim=-1)
    )
    nn.init.normal_(gate_B[0].weight, std=0.1)
    nn.init.normal_(gate_B[1].weight, std=0.5)
    gate_B = gate_B.to(device)
    
    gate_a_params = sum(p.numel() for p in gate_A.parameters())
    gate_b_params = sum(p.numel() for p in gate_B.parameters())
    model_id = id(model)
    
    print(f"  🔧 Created DUAL independent gates:  {task_num} tasks → {expert_num} experts")
    print(f"      Model ID: {model_id}")
    print(f"      Gate A (gate_up_proj) params: {gate_a_params:,}")
    print(f"      Gate B (down_proj) params:    {gate_b_params:,}")
    print(f"      Temperature: {temperature}")
    print(f"      Device: {device}")
    
    # ✅ 关键1：将两个门控注册为模型的直接子模块
    model.global_task_gate_A = gate_A
    model.global_task_gate_B = gate_B
    
    # ✅ 关键2：注册到全局字典（使用模型 ID）
    _DUAL_GATE_REGISTRY[model_id] = {
        'gate_A': gate_A,
        'gate_B': gate_B
    }
    
    replaced_count = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # ✅ 只存储模型 ID，不存储模型引用
            mlp._root_model_id = model_id
            
            for proj_name in ['gate_up_proj', 'down_proj']:
                if not hasattr(mlp, proj_name):
                    continue
                
                old_proj = getattr(mlp, proj_name)
                if not isinstance(old_proj, nn.Linear):
                    continue
                
                # 创建新的 MMOELoraLinearS
                new_proj = MMOELoraLinearS(
                    adapter_name=adapter_name,
                    in_features=old_proj.in_features,
                    out_features=old_proj.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    fan_in_fan_out=False,
                    init_lora_weights=True,
                    bias=(old_proj.bias is not None),
                    expert_num=expert_num
                )
                
                # 复制原始权重
                with torch.no_grad():
                    new_proj.weight.copy_(old_proj.weight)
                    if hasattr(old_proj, 'bias') and old_proj.bias is not None:
                        new_proj.bias.copy_(old_proj.bias)
                
                new_proj = new_proj.to(device)
                
                # ✅ 存储元信息（不存储门控本身）
                new_proj.task_num = task_num
                new_proj.task_embedding_dim = task_embedding_dim
                new_proj.expert_num = expert_num
                new_proj._layer_idx = layer_idx
                new_proj._proj_name = proj_name
                
                setattr(mlp, proj_name, new_proj)
                replaced_count += 1
                
                if layer_idx < 2:
                    gate_type = "Gate A" if proj_name == 'gate_up_proj' else "Gate B"
                    print(f"    Layer {layer_idx}.{proj_name}: Created (will use {gate_type})")
    
    print(f"\n🎉 Total replaced: {replaced_count} linear layers")
    print(f"📌 Gate A & B independently trained ({gate_a_params + gate_b_params:,} total gate params)")
    
    return model


def patch_phi3mlp_forward():
    """
    修改 Phi3MLP 的 forward 方法以支持 MMOELoraLinearS（双门控）
    
    ✅ gate_up_proj 使用 gate_A
    ✅ down_proj 使用 gate_B
    ✅ 通过全局注册表获取门控，避免循环引用
    """
    
    original_forward = Phi3MLP.forward
    
    def moelora_forward(self, hidden_state, **kwargs):
        """
        新的 forward 方法，支持 MMOELoraLinearS + 双门控
        """
        # 获取 task_id
        task_id = kwargs.get('task_id', None)
        if task_id is None:
            task_id = getattr(self, '_task_id', None)
        
        # ✅ 通过全局注册表获取两个门控（使用模型 ID）
        gate_A = None
        gate_B = None
        if hasattr(self, '_root_model_id'):
            model_id = self._root_model_id
            gates = _DUAL_GATE_REGISTRY.get(model_id, None)
            if gates:
                gate_A = gates['gate_A']
                gate_B = gates['gate_B']
        
        # ✅ gate_up_proj 使用 gate_A
        if isinstance(self.gate_up_proj, MMOELoraLinearS):
            gate_up_out = self.gate_up_proj(
                hidden_state, 
                task_id=task_id,
                global_gate=gate_A  # ✅ 传递 gate_A
            )
        else:
            gate_up_out = self.gate_up_proj(hidden_state)
        
        # Split for gating
        gate_proj, up_proj = gate_up_out.chunk(2, dim=-1)
        
        # Activation
        intermediate = self.activation_fn(gate_proj) * up_proj
        
        # ✅ down_proj 使用 gate_B
        if isinstance(self.down_proj, MMOELoraLinearS):
            down_out = self.down_proj(
                intermediate, 
                task_id=task_id,
                global_gate=gate_B  # ✅ 传递 gate_B
            )
        else:
            down_out = self.down_proj(intermediate)
        
        if down_out.dtype != hidden_state.dtype:
            down_out = down_out.to(hidden_state.dtype)
        return down_out
    
    Phi3MLP.forward = moelora_forward
    print("✅ Patched Phi3MLP.forward() to support MMOELoraLinearS with dual independent gates")


def verify_replacement(model):
    """验证 MLP 层是否成功替换"""
    print("\n🔍 Verifying MLP replacement...")
    
    replaced_count = 0
    total_count = 0
    
    for layer in model.model.layers:
        if hasattr(layer, 'mlp'):
            for proj_name in ['gate_up_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    total_count += 1
                    proj = getattr(layer.mlp, proj_name)
                    
                    if isinstance(proj, MMOELoraLinearS):
                        replaced_count += 1
    
    print(f"  Replaced layers:         {replaced_count}/{total_count}")
    
    if replaced_count == total_count:
        print("  ✅ All projections successfully replaced!")
        
        if hasattr(model, 'global_task_gate_A') and hasattr(model, 'global_task_gate_B'):
            print(f"  ✅ Dual gates registered")
        
        model_id = id(model)
        if model_id in _DUAL_GATE_REGISTRY:
            gates = _DUAL_GATE_REGISTRY[model_id]
            print(f"  ✅ Gates registered in global registry (ID: {model_id})")
            print(f"     ├─ Gate A registered")
            print(f"     └─ Gate B registered")
        
        return True
    else:
        print(f"  ⚠️ Warning:  Replacement incomplete")
        return False


def verify_gate_sharing(model):
    """验证双门控配置"""
    print("\n🔍 Verifying dual gate configuration...")
    
    if not (hasattr(model, 'global_task_gate_A') and hasattr(model, 'global_task_gate_B')):
        print("  ❌ No dual gates found!")
        return False
    
    model_id = id(model)
    gate_a_params = sum(p.numel() for p in model.global_task_gate_A.parameters())
    gate_b_params = sum(p.numel() for p in model.global_task_gate_B.parameters())
    
    # 检查 MLP 是否有模型 ID 引用
    mlp_with_id = 0
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_root_model_id'):
            mlp_with_id += 1
    
    print(f"  ✅ MLPs with model ID reference: {mlp_with_id}/{len(model.model.layers)}")
    
    # 检查前几层
    for layer_idx in [0, 1, model.model.config.num_hidden_layers - 1]:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'mlp'):
            print(f"    Layer {layer_idx}:")
            if isinstance(layer.mlp.gate_up_proj, MMOELoraLinearS):
                print(f"      ✅ gate_up_proj: Uses Gate A (independent)")
            if isinstance(layer.mlp.down_proj, MMOELoraLinearS):
                print(f"      ✅ down_proj: Uses Gate B (independent)")
    
    print(f"\n  Summary:")
    print(f"    Gate A params: {gate_a_params:,}")
    print(f"    Gate B params: {gate_b_params:,}")
    print(f"    Total gate params: {gate_a_params + gate_b_params:,}")
    print(f"  ✅ Dual gate configuration correct!")
    
    return True


def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params += param.numel()
    
    # ✅ 分别统计两个门控
    gate_a_params = 0
    gate_b_params = 0
    if hasattr(model, 'global_task_gate_A'):
        gate_a_params = sum(p.numel() for p in model.global_task_gate_A.parameters() if p.requires_grad)
    if hasattr(model, 'global_task_gate_B'):
        gate_b_params = sum(p.numel() for p in model.global_task_gate_B.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'lora': lora_params,
        'gate_a': gate_a_params,
        'gate_b': gate_b_params,
        'gate_total': gate_a_params + gate_b_params,
        'ratio': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_parameter_stats(model):
    """打印参数统计信息"""
    stats = count_parameters(model)
    
    print("\n" + "="*60)
    print("📊 Parameter Statistics (Dual Gate):")
    print(f"  Total params:              {stats['total']:>15,}")
    print(f"  Trainable params:         {stats['trainable']:>15,}")
    print(f"    ├─ LoRA params:         {stats['lora']:>15,}")
    print(f"    ├─ Gate A params:       {stats['gate_a']:>15,}")
    print(f"    └─ Gate B params:       {stats['gate_b']:>15,}")
    print(f"  Total gate params:        {stats['gate_total']:>15,}")
    print(f"  Trainable ratio:          {stats['ratio']:>14.4f}%")
    print("="*60)
