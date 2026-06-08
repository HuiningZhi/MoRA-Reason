"""
扩展的 Trainer，支持 MOELoRA 的 task_id 传递和 safetensors 分片保存
"""
import os
import torch
from . tinyllava_trainer import LLaVATrainer
import numpy as np
import sys


class MOELoRATrainer(LLaVATrainer):
    """
    扩展的 Trainer，支持：
    1. 在每次前向传播前将 task_ids 设置到 MLP 层
    2. 使用 safetensors 格式分片保存（model-*. safetensors）
    """
    
    def __init__(self, *args, **kwargs):
        train_dataset_ref = kwargs.get('train_dataset', None)
        super().__init__(*args, **kwargs)
        print("✅ Using MOELoRATrainer with task_id support and safetensors saving")
        # 保存 train_dataset
        if train_dataset_ref is not None:
            self.train_dataset = train_dataset_ref
        elif hasattr(self, 'train_dataset'):
            pass  # 父类已设置
        else:
            self.train_dataset = None
        # ✅ 验证
        if self.train_dataset is not None:
            print(f"   ✅ Train dataset: {len(self.train_dataset)} samples")
            print(f"   ✅ Dataset type: {type(self.train_dataset)}")
            
            # 测试访问第一个样本
            try:
                test_sample = self.train_dataset. list_data_dict[0]
                print(f"   ✅ Test access: {list(test_sample.keys())}")
            except Exception as e: 
                print(f"   ⚠️ Cannot access list_data_dict: {e}")
        else:
            print(f"   ❌ Train dataset is None!")
        # ✅ 初始化门控统计
        self._last_print_step = -1
    
    def _save(self, output_dir=None, state_dict=None):
        """
        重写保存方法 - 强制使用 safetensors 格式
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving model to {output_dir}")
        print(f"   Format: safetensors (sharded)")
        
        try:
            # 尝试直接保存
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            print(f"✅ Saved with safetensors format")
        
        except RuntimeError as e:
            if "shared tensors" in str(e).lower():
                print(f"⚠️ Shared tensor detected, cleaning state_dict...")
                # 清理共享张量后重试
                self._save_with_cleaned_state_dict(output_dir, state_dict)
            else:
                raise
        
        # 保存 tokenizer 和训练参数
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print(f"✅ Checkpoint saved to {output_dir}")
    
    def _save_with_cleaned_state_dict(self, output_dir, state_dict=None):
        """
        清理共享门控后保存 - 使用 safetensors 格式
        
        策略：移除所有层内的 _global_gate 引用，只保留主 global_task_gate
        """
        print("   Cleaning shared gate references...")
        
        # 获取 state_dict
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # 找到所有门控相关的键
        all_keys = list(state_dict.keys())
        
        # 创建清理后的 state_dict
        cleaned_state_dict = {}
        removed_count = 0
        kept_count = 0
        
        for key, value in state_dict.items():
            # 检查是否是层内的 _global_gate（需要移除）
            if '._global_gate.' in key:
                # 这是层内的门控引用，跳过
                removed_count += 1
                continue
            else:
                # 保留所有其他参数（包括主 global_task_gate）
                cleaned_state_dict[key] = value
                kept_count += 1
        
        print(f"   Original keys: {len(state_dict)}")
        print(f"   Removed keys:   {removed_count} (layer _global_gate references)")
        print(f"   Kept keys:     {kept_count}")
        
        # 验证主门控是否保留
        main_gate_keys = [k for k in cleaned_state_dict.keys() if 'global_task_gate' in k and '._global_gate' not in k]
        if main_gate_keys:
            print(f"   ✅ Main gate preserved: {len(main_gate_keys)} keys")
            for key in main_gate_keys: 
                print(f"      - {key}")
        else:
            print(f"   ⚠️ Warning: No main gate keys found!")
        
        # 使用清理后的 state_dict 保存
        try:
            self.model.save_pretrained(
                output_dir,
                state_dict=cleaned_state_dict,
                safe_serialization=True,      # ✅ 使用 safetensors
                max_shard_size="5GB"
            )
            
            print(f"   ✅ Saved with safetensors (cleaned)")
            
            # 验证保存的文件
            saved_files = os.listdir(output_dir)
            safetensors_files = [f for f in saved_files if f.endswith('.safetensors') and f.startswith('model-')]
            
            if safetensors_files:
                print(f"   ✅ Verified safetensors files:")
                for f in sorted(safetensors_files):
                    size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024*1024)
                    print(f"      - {f} ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"   ❌ Failed even after cleaning: {e}")
            # 如果还是失败，打印详细信息用于调试
            print(f"\n   Debug:  Checking remaining gate keys...")
            remaining_gate_keys = [k for k in cleaned_state_dict.keys() if 'gate' in k. lower()]
            print(f"   Found {len(remaining_gate_keys)} keys with 'gate':")
            for key in remaining_gate_keys[: 10]: 
                print(f"      - {key}")
            if len(remaining_gate_keys) > 10:
                print(f"      ... and {len(remaining_gate_keys) - 10} more")
            raise
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss，在前向传播前设置 task_ids
        """
        # 提取 task_ids
        task_ids = inputs.pop('task_ids', None)
        answer_types = inputs.pop('answer_type', None)
        conversations = inputs.pop('conversations', None) 
        sample_indices = inputs.pop('sample_indices', None)
        if task_ids is None:
            print("❌ ERROR: No task_ids in batch!This should never happen if Dataset is working correctly. Check Dataset.__getitem__ and DataCollator implementation.")
        
        # ✅ 【添加这一整段】获取 LLM（用于后续的打印和统计）
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        llm = None
        if hasattr(actual_model, 'language_model'):
            llm = actual_model.language_model
        elif hasattr(actual_model, 'llm'):
            llm = actual_model.llm
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            llm = actual_model
        
        
        
        
        # ✅ 修改5：初始化计数器
        if not hasattr(self, '_sample_log_count'):
            self._sample_log_count = 0
        
        # ✅ 修改6：打印前 10 个样本的验证信息
        if self._sample_log_count < 10:
            # 转换
            if isinstance(task_ids, torch.Tensor):
                task_ids_list = task_ids.cpu().tolist()
            else:
                task_ids_list = task_ids
            
            if isinstance(sample_indices, torch. Tensor):
                indices_list = sample_indices.cpu().tolist()
            else:
                indices_list = sample_indices if sample_indices is not None else [-1] * len(task_ids_list)
            
            

            
            # ✅ 修改7：打印每个样本的验证信息
            for idx, (tid, atype, sample_idx) in enumerate(zip(task_ids_list, answer_types, indices_list)):
                print(f"\n{'='*80}")
                print(f"📋 Sample {self._sample_log_count + 1} - Gate Activation Verification")
                print(f"{'='*80}")
                print(f"Answer Type:    {atype}")
                print(f"Task ID:       {tid}")
                print(f"Sample Index:  {sample_idx}")
                
                # ✅ 修改8：通过 sample_idx 回查原始 sources
                if self.train_dataset is not None and sample_idx >= 0:
                    try: 
                        sources = self.train_dataset.list_data_dict[sample_idx]
                        print(f"\n📄 Original Data:")
                        print(sources)  # 打印完整的原始数据
                    except Exception as e:
                        print(f"   ⚠️ Error:  {e}")
                else:
                    print(f"   ⚠️ Cannot retrieve data (dataset={self.train_dataset is not None}, idx={sample_idx})")
                
                # ✅ 修改9：查询并显示门控���重
                if llm and hasattr(llm, 'global_task_gate'):
                    gate_device = next(llm.global_task_gate.parameters()).device
                    task_tensor = torch.tensor([tid], dtype=torch.long, device=gate_device)
                    
                    with torch.no_grad():
                        expert_weights = llm.global_task_gate(task_tensor)
                        weights = expert_weights[0].cpu().numpy()
                        selected_expert = expert_weights[0].argmax().item()
                        confidence = weights[selected_expert]
                        
                        print(f"\n🎯 Gate:  Expert {selected_expert} ({confidence:.2%})")
                        print(f"   {weights}")
                        
                        # 计算专门化程度
                        
                        entropy = -np.sum(weights * np.log(weights + 1e-10))
                        max_entropy = -np.log(1.0 / len(weights))
                        specialization = 1 - (entropy / max_entropy)
                        
                        print(f"✅ Specialization: {specialization:.2%}")
                
                print(f"{'='*80}\n")
                
                self._sample_log_count += 1
                if self._sample_log_count >= 10:
                    break
        # ✅ 【添加这一整段】每 100 步打印统计
        current_step = self.state.global_step
        if current_step > 0 and current_step % 100 == 0 and current_step != self._last_print_step:
            self._print_current_step_weights(llm, task_ids, answer_types, current_step)
            self._last_print_step = current_step       
        # ✅ 修改10：设置 task_ids 到设备
        device = next(model.parameters()).device
        if not isinstance(task_ids, torch.Tensor):
            task_ids = torch.tensor(task_ids, dtype=torch.long)
        task_ids = task_ids.to(device)
        
        # ✅ 修改11：找到 LLM 并设置 task_ids
        if llm and hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            for layer in llm.model. layers:
                if hasattr(layer, 'mlp'):
                    layer.mlp._task_id = task_ids
        
        # ✅ 修改12：调用父类
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def _print_current_step_weights(self, llm, task_ids, answer_types, step):
        """
        打印当前 step 的门控权重（不累积，只看当前值）
        ✅ 始终打印 OPEN 和 CLOSED 的权重，不管当前 batch 是否包含
        
        Args:
            llm:  语言模型
            task_ids: 当前 batch 的 task_ids
            answer_types: 当前 batch 的 answer_types
            step: 当前 step 编号
        """
        if llm is None or not hasattr(llm, 'global_task_gate'):
            return
        
        sys.stdout.flush()
        
        print(f"\n{'='*80}")
        print(f"📊 GATE STATISTICS - Step {step}")
        print(f"{'='*80}")
        
        # ✅ 统计当前 batch 的样本分布（仅用于显示）
        if isinstance(task_ids, torch.Tensor):
            task_ids_list = task_ids.cpu().tolist()
        else:
            task_ids_list = list(task_ids)
        
        open_count = sum(1 for tid in task_ids_list if tid == 0)
        closed_count = sum(1 for tid in task_ids_list if tid == 1)
        
        print(f"Current batch:  {open_count} OPEN samples, {closed_count} CLOSED samples")
        
        # ✅ 始终打印 OPEN 和 CLOSED 的权重
        task_mapping = {0: 'OPEN', 1: 'CLOSED'}
        gate_device = next(llm.global_task_gate. parameters()).device
        
        with torch.no_grad():
            for task_id, task_name in task_mapping.items():
                # ✅ 获取这个任务的门控权重（当前时刻的值）
                task_tensor = torch.tensor([task_id], dtype=torch.long, device=gate_device)
                expert_weights = llm.global_task_gate(task_tensor)
                weights = expert_weights[0].cpu().numpy()
                
                # 找到最大权重的专家
                max_expert = weights.argmax()
                max_weight = weights[max_expert]
                
                # 计算专门化程度
                entropy = -np.sum(weights * np.log(weights + 1e-10))
                max_entropy = -np.log(1.0 / len(weights))
                specialization = 1 - (entropy / max_entropy)
                
                # ✅ 统计当前 batch 中这个任务的样本数（如果有的话）
                count_in_batch = sum(1 for tid in task_ids_list if tid == task_id)
                batch_info = f" (n={count_in_batch} in current batch)" if count_in_batch > 0 else " (not in current batch)"
                
                print(f"\n🎯 {task_name}{batch_info}:")
                print(f"   Gate Weights:  {weights}")
                print(f"   Most Selected: Expert {max_expert} ({max_weight:.2%})")
                
                # 可视化
                for i, weight in enumerate(weights):
                    bar = "█" * int(weight * 40)
                    marker = " ⭐" if i == max_expert else ""
                    print(f"     Expert {i}:  {weight:.4f} |{bar: <40}|{marker}")
                
                print(f"   Specialization:  {specialization:.2%}")
        
        # 对比 OPEN 和 CLOSED（当前时刻的值）
        print(f"\n📊 Comparison:")
        with torch.no_grad():
            open_tensor = torch.tensor([0], dtype=torch.long, device=gate_device)
            closed_tensor = torch.tensor([1], dtype=torch.long, device=gate_device)
            
            open_weights = llm.global_task_gate(open_tensor)[0].cpu().numpy()
            closed_weights = llm.global_task_gate(closed_tensor)[0].cpu().numpy()
            
            open_max = open_weights.argmax()
            closed_max = closed_weights.argmax()
            
            print(f"   OPEN   → Expert {open_max} ({open_weights[open_max]:.2%})")
            print(f"   CLOSED → Expert {closed_max} ({closed_weights[closed_max]:.2%})")
            
            # 计算权重分布的差异
            diff = np.abs(open_weights - closed_weights).sum()
            print(f"   Difference: {diff:.4f}")
        
        print(f"{'='*80}\n")
        sys.stdout.flush()