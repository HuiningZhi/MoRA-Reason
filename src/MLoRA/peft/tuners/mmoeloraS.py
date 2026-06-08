"""
MMOELoraLinearS - 多专家混合 LoRA（支持全局门控）
✅ DeepSpeed 兼容版本：通过 forward 参数传递全局门控
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from dataclasses import dataclass, field


class MMOELoraLinearS(nn.Module):
    """
    多专家混合 LoRA Linear 层
    ✅ 支持全局共享门控（通过参数传递，避免梯度重复计算）
    """
    
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha:  int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        bias: bool = True,
        expert_num: int = 4,
        **kwargs
    ):
        """初始化 MMOELoraLinearS"""
        super().__init__()
        
        # 基础 Linear 层
        self. in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化基础权重
        nn. init.kaiming_uniform_(self.weight, a=math. sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn. init._calculate_fan_in_and_fan_out(self. weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # LoRA 配置
        self.fan_in_fan_out = fan_in_fan_out
        self. expert_num = expert_num
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        
        # 为每个专家创建 LoRA A 和 B
        if r > 0:
            self. lora_A = nn.ModuleList([
                nn.Linear(in_features, r, bias=False) for _ in range(expert_num)
            ])
            self.lora_B = nn.ModuleList([
                nn.Linear(r, out_features, bias=False) for _ in range(expert_num)
            ])
            
            # Dropout
            if lora_dropout > 0.0:
                self.lora_dropout = nn. Dropout(p=lora_dropout)
            else:
                self.lora_dropout = nn.Identity()
            
            # 初始化 LoRA 权重
            if init_lora_weights: 
                for expert_idx in range(expert_num):
                    nn.init.kaiming_uniform_(self.lora_A[expert_idx].weight, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[expert_idx].weight)
        
        # 状态标志
        self.merged = False
        self.disable_adapters = False
        
        # ✅ 不存储全局门控的引用
        # 门控将通过 forward 的 global_gate 参数传递
        self._global_gate = None  # 保留占位符用于兼容性
        
        # 适配器信息
        self.active_adapter = adapter_name
        self.active_adapters = [adapter_name]
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_id:  Optional[Union[int, torch.Tensor]] = None,
        global_gate: Optional[nn.Module] = None,  # ✅ 新增参数
        *args, 
        **kwargs
    ):
        """
        前向传播
        
        ✅ 修复版：接收 global_gate 作为参数，避免共享参数梯度问题
        
        Args: 
            x: 输入张量 [batch_size, seq_len, in_features]
            task_id: 任务 ID（int 或 tensor）
            global_gate: 全局门控模块（通过参数传递）
        
        Returns:
            输出张量 [batch_size, seq_len, out_features]
        """
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        # 基础 Linear 输出
        result = F.linear(x, self.weight, self.bias)
        
        # 如果禁用适配器或已合并，直接返回
        if self.disable_adapters or self.merged or self.r == 0:
            return result
        
        # ✅ 如果没有提供 global_gate，尝试使用存储的引用（向后兼容）
        if global_gate is None:
            global_gate = self._global_gate
        
        # 获取专家权重
        expert_weights = self._get_expert_weights(x, task_id, global_gate)
        
        # 应用 LoRA
        lora_output = self._apply_lora(x, expert_weights)
        
        return result + lora_output
    
    def _get_expert_weights(self, x, task_id, global_gate):
        """
        获取专家权重
        
        ✅ 修复版：使用传入的 global_gate
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 使用全局门控
        if global_gate is not None and task_id is not None:
            # 确保 task_id 是正确格式
            if not isinstance(task_id, torch.Tensor):
                task_id = torch. tensor([task_id], dtype=torch.long, device=device)
            else:
                task_id = task_id.to(device)
            
            if task_id. dim() == 0:
                task_id = task_id. unsqueeze(0)
            
            # 扩展 task_id 以匹配 batch_size
            if task_id.shape[0] == 1 and batch_size > 1:
                task_id = task_id.expand(batch_size)
            
            # ✅ 调用传入的全局门控（保持梯度追踪）
            with torch.set_grad_enabled(self.training):
                expert_weights = global_gate(task_id)  # [batch_size, expert_num]
        else:
            # 默认均匀分配
            expert_weights = torch.ones(batch_size, self.expert_num, device=device) / self.expert_num
        
        return expert_weights
    
    def _apply_lora(self, x, expert_weights):
        """应用 LoRA 变换"""
        # 为每个专家计算输出
        expert_outputs = []
        for expert_idx in range(self.expert_num):
            lora_out = self.lora_B[expert_idx](
                self.lora_A[expert_idx](self.lora_dropout(x))
            )
            expert_outputs.append(lora_out)
        
        # Stack:  [expert_num, batch_size, seq_len, out_features]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # 扩展权重维度: [expert_num, batch_size, 1, 1]
        expert_weights_expanded = expert_weights.t().unsqueeze(-1).unsqueeze(-1)
        
        # 加权求和
        weighted_output = (expert_outputs * expert_weights_expanded).sum(dim=0)
        
        return weighted_output * self.scaling
    
    def __repr__(self):
        return (
            f"MMOELoraLinearS(\n"
            f"  in_features={self.in_features},\n"
            f"  out_features={self.out_features},\n"
            f"  r={self.r},\n"
            f"  lora_alpha={self.lora_alpha},\n"
            f"  expert_num={self.expert_num},\n"
            f"  scaling={self.scaling:. 4f}\n"
            f")"
        )
    
    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'r={self.r}, '
            f'expert_num={self.expert_num}'
        )


# ========================================
# PEFT 配置类（用于兼容性）
# ========================================

try:
    from peft. config import PeftConfig
    from peft.utils import PeftType
    
    @dataclass
    class MMOELoraConfigS(PeftConfig):
        """MMOELoRA 配置类"""
        r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
        target_modules: Optional[Union[list, str]] = field(
            default=None,
            metadata={"help": "List of module names to apply LoRA"}
        )
        lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
        lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
        fan_in_fan_out: bool = field(
            default=False,
            metadata={"help": "Set to True if layer uses Conv1D"}
        )
        bias: str = field(
            default="none",
            metadata={"help": "Bias type for LoRA.  Can be 'none', 'all' or 'lora_only'"}
        )
        expert_num: int = field(default=4, metadata={"help": "Number of experts"})
        task_num: int = field(default=2, metadata={"help": "Number of tasks"})
        task_embedding_dim: int = field(
            default=32,
            metadata={"help": "Task embedding dimension"}
        )
        
        def __post_init__(self):
            self.peft_type = PeftType. LORA
    
    from peft.tuners.lora import LoraModel
    
    class MMOELoraModelS(LoraModel):
        """MMOELoRA 模型类"""
        
        def __init__(self, model, config, adapter_name):
            super().__init__(model, config, adapter_name)
        
        @staticmethod
        def _create_new_module(lora_config, adapter_name, target, **kwargs):
            """创建新的 MMOELoRA 模块"""
            if isinstance(target, nn.Linear):
                new_module = MMOELoraLinearS(
                    adapter_name=adapter_name,
                    in_features=target.in_features,
                    out_features=target.out_features,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    fan_in_fan_out=lora_config.fan_in_fan_out,
                    bias=target.bias is not None,
                    expert_num=lora_config.expert_num,
                )
                
                # 复制原始权重
                with torch.no_grad():
                    new_module.weight.copy_(target.weight)
                    if target.bias is not None:
                        new_module.bias. copy_(target.bias)
                
                return new_module
            else:
                raise ValueError(
                    f"Target module {target.__class__.__name__} is not supported.  "
                    "Only nn.Linear is supported."
                )

except ImportError:
    # 如果 PEFT 未安装，跳过配置类
    print("⚠️  PEFT not installed, MMOELoraConfigS and MMOELoraModelS not available")
    pass