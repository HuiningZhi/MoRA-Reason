"""
MOELoRA 模型评估脚本
"""
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math

def load_moelora_model(model_path, expert_num=4, rank=16, alpha=16, device_map="auto"):
    """
    加载 MOELoRA 模型
    
    Args:
        model_path: 模型检查点路径
        expert_num: 专家数量
        rank: LoRA rank (对外接口使用 rank)
        alpha: LoRA alpha (对外接口使用 alpha)
        device_map: 设备映射
    
    Returns:
        model, tokenizer, image_processor, context_len
    """
    print(f"\n{'='*80}")
    print(f"📂 Loading MOELoRA Model")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    print(f"Expert num: {expert_num}, Rank: {rank}, Alpha: {alpha}")
    
    # ========================================
    # 1. 加载基础模型结构
    # ========================================
    print(f"\n[1/5] Loading base model...")
    from tinyllava.model import load_pretrained_model
    
    model, tokenizer, image_processor, context_len = load_pretrained_model(
        model_path,
        expert_num=expert_num,
        lora_r=rank,
        lora_alpha=alpha
    )
    
    print(f"      ✅ Base model loaded")
    
    # ========================================
    # 2. Patch Phi3MLP forward
    # ========================================
    print(f"\n[2/5] Patching Phi3MLP forward...")
    from tinyllava.train.custom_finetune_moelora import patch_phi3mlp_forward
    patch_phi3mlp_forward()
    print(f"      ✅ Phi3MLP forward patched")
    
    # ========================================
    # 3. 应用 MOELoRA 替换
    # ========================================
    print(f"\n[3/5] Applying MOELoRA structure...")
    
    from tinyllava.train.custom_finetune_moelora import replace_phi3mlp_with_mmoeloraS
    
    llm_model = model.language_model
    # ✅ 参数名转换：rank → lora_r, alpha → lora_alpha
    llm_model = replace_phi3mlp_with_mmoeloraS(
        llm_model,
        lora_r=rank,              # ✅ 内部使用 lora_r
        lora_alpha=alpha,         # ✅ 内部使用 lora_alpha
        lora_dropout=0.1,
        expert_num=expert_num,
        task_num=2,
        task_embedding_dim=32,
        adapter_name="default",
        device="cpu"
    )
    model.language_model = llm_model
    
    print(f"      ✅ MOELoRA structure applied")
    
    # ========================================
    # 4. 重新加载权重（包含 MOELoRA 参数）
    # ========================================
    print(f"\n[4/5] Loading MOELoRA weights...")
    
    import glob
    from safetensors import safe_open
    
    # 查找 safetensors 文件
    safetensor_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
    
    if safetensor_files: 
        print(f"      Found {len(safetensor_files)} safetensors files")
        
        # 加载所有分片
        state_dict = {}
        for shard_file in sorted(safetensor_files):
            print(f"      Loading {os.path.basename(shard_file)}...")
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        
        print(f"      Loaded {len(state_dict)} parameters")
        
        # 加载到模型
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"      ✅ Weights loaded")
        if missing_keys: 
            print(f"         Missing keys: {len(missing_keys)}")
        if unexpected_keys: 
            print(f"         Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"      ⚠️ No safetensors files found, checking for pytorch_model files...")
        # 尝试加载 pytorch bin 文件
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
        if bin_files:
            state_dict = {}
            for bin_file in sorted(bin_files):
                print(f"      Loading {os.path.basename(bin_file)}...")
                shard_dict = torch.load(bin_file, map_location='cpu')
                state_dict.update(shard_dict)
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"      ✅ Weights loaded from pytorch_model files")
        else:
            print(f"      ❌ No model weight files found!")
            raise FileNotFoundError("No model weights found")
    
    # ========================================
    # 5. 验证 MOELoRA 组件
    # ========================================
    print(f"\n[5/5] Verifying MOELoRA components...")
    
    from src.MLoRA.peft.tuners.mmoeloraS import MMOELoraLinearS
    
    # ✅ 检查双门控
    if hasattr(model.language_model, 'global_task_gate_A') and hasattr(model.language_model, 'global_task_gate_B'):
        gate_a_params = sum(p.numel() for p in model.language_model.global_task_gate_A.parameters())
        gate_b_params = sum(p.numel() for p in model.language_model.global_task_gate_B.parameters())
        print(f"      ✅ Dual gates found: Gate A ({gate_a_params:,} params) + Gate B ({gate_b_params:,} params)")
    else:
        print(f"      ❌ Dual gates NOT found!")
    
    # 检查 MOELoRA 层
    if hasattr(model.language_model.model, 'layers'):
        first_layer = model.language_model.model.layers[0]
        if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'gate_up_proj'):
            proj = first_layer.mlp.gate_up_proj
            if isinstance(proj, MMOELoraLinearS):
                print(f"      ✅ MOELoRA layers verified")
                print(f"         LoRA rank: {proj.r}")
                print(f"         Expert num: {proj.expert_num}")
                print(f"         Alpha: {proj.lora_alpha}")
            else:
                print(f"      ❌ Layer 0 is {type(proj).__name__}, not MMOELoraLinearS!")
    
    # ========================================
    # 6. 设置为评估模式并移动到设备
    # ========================================
    print(f"\n[6/6] Finalizing...")
    model.eval()
    
    # 如果需要移动到 GPU
    if device_map == "auto" or "cuda" in str(device_map):
        model = model.to('cuda')
        print(f"      ✅ Model moved to CUDA")
    
    print(f"\n{'='*80}")
    print(f"✅ MOELORA MODEL LOADED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Model:              {type(model).__name__}")
    print(f"Tokenizer:          {type(tokenizer).__name__}")
    print(f"Image Processor:    {type(image_processor).__name__}")
    print(f"Context Length:     {context_len}")
    print(f"Device:             {next(model.parameters()).device}")
    print(f"{'='*80}\n")
    
    return model, tokenizer, image_processor, context_len


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i: i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        answer_type = line.get("answer_type", "OPEN")  # 默认 OPEN
        answer_type_str = str(answer_type).upper()
        if answer_type_str in ['CLOSED', 'CLOSE']:
            task_id = 1
        else:
            task_id = 0  # OPEN
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size, answer_type_str, task_id

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, answer_types, task_ids = zip(*batch)  
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    task_ids = torch.tensor(task_ids, dtype=torch.long)
    return input_ids, image_tensors, image_sizes, answer_types, task_ids


def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    """评估 MOELoRA 模型"""
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # ========================================
    # 加载 MOELoRA 模型
    # ========================================
    model, tokenizer, image_processor, context_len = load_moelora_model(
        model_path,
        expert_num=args.expert_num,
        rank=args.rank,
        alpha=args.alpha
    )
    
    # 转换数据类型
    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)
    
    # 准备数据处理器
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
    
    # 推理
    model.to(device='cuda')
    model.eval()
    
    # ========================================
    # ✅ 任务映射和统计
    # ========================================
    task_mapping = {
        'OPEN': 0,
        'CLOSED': 1
    }
    
    task_stats = {'OPEN': 0, 'CLOSED': 0, 'UNKNOWN': 0}
    
    # ========================================
    # ✅ 双门控权重可视化（推理前）
    # ========================================
    print(f"\n{'='*80}")
    print(f"🎯 MOELoRA Dual Gate Expert Selection Verification (测试)")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for task_name, task_id in task_mapping.items():
            task_tensor = torch.tensor([task_id], dtype=torch.long, device='cuda')
            
            print(f"\n{task_name} (Task ID={task_id}):")
            
            # Gate A (用于 gate_up_proj)
            if hasattr(model.language_model, 'global_task_gate_A'):
                expert_weights_a = model.language_model.global_task_gate_A(task_tensor)
                print(f"  Gate A (gate_up_proj):")
                print(f"    Expert weights: {expert_weights_a[0].cpu().numpy()}")
                
                max_expert = expert_weights_a[0].argmax().item()
                for i, weight in enumerate(expert_weights_a[0].cpu().numpy()):
                    bar = "█" * int(weight * 40)
                    marker = " ⭐" if i == max_expert else ""
                    print(f"      Expert {i}: {weight:.4f} |{bar}|{marker}")
            
            # Gate B (用于 down_proj)
            if hasattr(model.language_model, 'global_task_gate_B'):
                expert_weights_b = model.language_model.global_task_gate_B(task_tensor)
                print(f"  Gate B (down_proj):")
                print(f"    Expert weights: {expert_weights_b[0].cpu().numpy()}")
                
                max_expert = expert_weights_b[0].argmax().item()
                for i, weight in enumerate(expert_weights_b[0].cpu().numpy()):
                    bar = "█" * int(weight * 40)
                    marker = " ⭐" if i == max_expert else ""
                    print(f"      Expert {i}: {weight:.4f} |{bar}|{marker}")
    
    print(f"\n{'='*80}\n")
    
    # ========================================
    # 推理循环
    # ========================================
    print(f"🚀 Starting inference on {len(questions)} questions...")
    
    # ✅ 解包时包含 answer_types 和 task_ids
    for idx, ((input_ids, image_tensor, image_sizes, answer_types_batch, task_ids_batch), line) in enumerate(
        tqdm(zip(data_loader, questions), total=len(questions))
    ):
        question_id = line["question_id"]
        cur_prompt = line["text"]
        
        # ✅ 从 batch 中获取（batch_size=1，所以取第一个）
        answer_type = answer_types_batch[0]
        task_id = task_ids_batch[0].item()
        task_id_tensor = task_ids_batch.to('cuda')
        
        # 统计
        task_stats[answer_type] = task_stats.get(answer_type, 0) + 1
        
        # ✅ 设置 task_id 到所有 MLP 层
        for layer in model.language_model.model.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp._task_id = task_id_tensor
        
        # ✅ 打印前几个样本的详细信息
        if idx < 10:
            print(f"\n{'='*60}")
            print(f"📝 Sample {idx+1}")
            print(f"{'='*60}")
            print(f"Question ID: {question_id}")
            print(f"Question: {cur_prompt[:100]}...")
            print(f"Answer Type: {answer_type}")
            print(f"Task ID: {task_id}")
            
            # 显示当前任务的专家权重（两个门控）
            with torch.no_grad():
                if hasattr(model.language_model, 'global_task_gate_A'):
                    expert_weights_a = model.language_model.global_task_gate_A(task_id_tensor)
                    print(f"Gate A weights: {expert_weights_a[0].cpu().numpy()}")
                
                if hasattr(model.language_model, 'global_task_gate_B'):
                    expert_weights_b = model.language_model.global_task_gate_B(task_id_tensor)
                    print(f"Gate B weights: {expert_weights_b[0].cpu().numpy()}")
        
        # ...  推理代码不变 ...
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_sizes=image_sizes,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        if idx < 5:
            print(f"Generated answer: {outputs[:150]}...")
            print(f"{'='*60}\n")
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": question_id,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": args.model_base,
            "metadata": {
                "answer_type": answer_type,
                "task_id": task_id
            }
        }) + "\n")
    
    ans_file.close()
    
    # ========================================
    # ✅ 最终统计
    # ========================================
    print(f"\n{'='*80}")
    print(f"✅ Inference Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {answers_file}")
    print(f"\n📊 Task Distribution:")
    for task_type, count in task_stats.items():
        percentage = (count / len(questions)) * 100
        print(f"{task_type:10s}: {count:4d} ({percentage:5.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    
    # MOELoRA 参数
    parser.add_argument("--expert_num", type=int, default=4, help="Number of experts")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    
    args = parser.parse_args()
    eval_model(args)
