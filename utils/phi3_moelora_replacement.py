"""
Replace Phi3 MLP projections with MMOELoraLinearS.

Each replaced projection owns an independent task gate:
- layer_i.mlp.gate_up_proj.task_gate
- layer_i.mlp.down_proj.task_gate

The input task_id is still shared across all projections for a sample, but every
projection learns its own task_id -> expert routing.
"""
import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3MLP

from src.MLoRA.peft.tuners.mmoeloraS import MMOELoraLinearS


class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1.0, dim=-1):
        super().__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, x):
        return nn.functional.softmax(x / self.temperature, dim=self.dim)


def create_task_gate(task_num, task_embedding_dim, expert_num, device="cpu", temperature=1.0):
    gate = nn.Sequential(
        nn.Embedding(task_num + 1, task_embedding_dim),
        nn.Linear(task_embedding_dim, expert_num, bias=False),
        TemperatureSoftmax(temperature=temperature, dim=-1),
    )
    nn.init.normal_(gate[0].weight, std=0.1)
    nn.init.normal_(gate[1].weight, std=0.5)
    return gate.to(device)


def iter_moelora_projections(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp"):
            continue
        for proj_name in ("gate_up_proj", "down_proj"):
            if not hasattr(layer.mlp, proj_name):
                continue
            proj = getattr(layer.mlp, proj_name)
            if isinstance(proj, MMOELoraLinearS):
                yield layer_idx, proj_name, proj


def get_moelora_gate(proj):
    if hasattr(proj, "task_gate"):
        return proj.task_gate
    if hasattr(proj, "_global_gate") and proj._global_gate is not None:
        return proj._global_gate
    return None


def is_moelora_gate_parameter_name(name):
    return ".task_gate." in name or "global_task_gate_A" in name or "global_task_gate_B" in name


def count_projection_gates(model):
    gate_up = 0
    down = 0
    params = 0
    for _, proj_name, proj in iter_moelora_projections(model):
        gate = get_moelora_gate(proj)
        if gate is None:
            continue
        if proj_name == "gate_up_proj":
            gate_up += 1
        elif proj_name == "down_proj":
            down += 1
        params += sum(p.numel() for p in gate.parameters())
    return {"gate_up": gate_up, "down": down, "total": gate_up + down, "params": params}


def print_moelora_gate_samples(model, title, task_mapping=None, max_layers=2):
    task_mapping = task_mapping or {"OPEN": 0, "CLOSED": 1}
    projections = list(iter_moelora_projections(model))
    if not projections:
        print(f"{title}: no MMOELoRA projections found")
        return

    selected = projections[: max_layers * 2]
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    with torch.no_grad():
        for task_name, task_id in task_mapping.items():
            print(f"\n{task_name} (Task ID={task_id}):")
            for layer_idx, proj_name, proj in selected:
                gate = get_moelora_gate(proj)
                if gate is None:
                    print(f"  Layer {layer_idx}.{proj_name}: no task_gate")
                    continue
                gate_device = next(gate.parameters()).device
                task_tensor = torch.tensor([task_id], dtype=torch.long, device=gate_device)
                weights = gate(task_tensor)[0].detach().cpu()
                max_expert = int(weights.argmax().item())
                weights_str = ", ".join(f"{float(w):.4f}" for w in weights)
                print(f"  Layer {layer_idx}.{proj_name}: [{weights_str}] -> expert {max_expert}")
    print("=" * 80 + "\n")


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
    temperature=1.0,
):
    """
    Replace Phi3MLP gate_up_proj/down_proj with MMOELoraLinearS.

    Unlike the previous dual-global-gate implementation, this creates one gate
    per projection. Layer 0 gate_up, layer 1 gate_up, layer 0 down, etc. all
    route independently while receiving the same task_id value.
    """
    print("\nReplacing Phi3MLP projections with MMOELoraLinearS (projection-local gates)...")

    replaced_count = 0
    gate_param_total = 0

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp

        for proj_name in ("gate_up_proj", "down_proj"):
            if not hasattr(mlp, proj_name):
                continue

            old_proj = getattr(mlp, proj_name)
            if isinstance(old_proj, MMOELoraLinearS):
                continue
            if not isinstance(old_proj, nn.Linear):
                continue

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
                expert_num=expert_num,
            )

            with torch.no_grad():
                new_proj.weight.copy_(old_proj.weight)
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)

            new_proj.task_gate = create_task_gate(
                task_num=task_num,
                task_embedding_dim=task_embedding_dim,
                expert_num=expert_num,
                device=device,
                temperature=temperature,
            )
            new_proj.task_num = task_num
            new_proj.task_embedding_dim = task_embedding_dim
            new_proj.expert_num = expert_num
            new_proj._layer_idx = layer_idx
            new_proj._proj_name = proj_name
            new_proj = new_proj.to(device)

            gate_param_total += sum(p.numel() for p in new_proj.task_gate.parameters())
            setattr(mlp, proj_name, new_proj)
            replaced_count += 1

            if layer_idx < 2:
                print(f"  Layer {layer_idx}.{proj_name}: local task_gate created")

    gate_stats = count_projection_gates(model)
    print(f"\nTotal replaced projections: {replaced_count}")
    print(
        "Projection-local gates: "
        f"gate_up={gate_stats['gate_up']}, down={gate_stats['down']}, "
        f"total={gate_stats['total']}, params={gate_param_total:,}"
    )
    print("Each projection now learns an independent task_id -> expert route.")

    return model


def patch_phi3mlp_forward():
    """
    Patch Phi3MLP.forward to pass task_id into MMOELoraLinearS projections.
    The projection itself owns the gate, so no global gate lookup is needed.
    """

    def moelora_forward(self, hidden_state, **kwargs):
        task_id = kwargs.get("task_id", None)
        if task_id is None:
            task_id = getattr(self, "_task_id", None)

        if isinstance(self.gate_up_proj, MMOELoraLinearS):
            gate_up_out = self.gate_up_proj(hidden_state, task_id=task_id)
        else:
            gate_up_out = self.gate_up_proj(hidden_state)

        gate_proj, up_proj = gate_up_out.chunk(2, dim=-1)
        intermediate = self.activation_fn(gate_proj) * up_proj

        if isinstance(self.down_proj, MMOELoraLinearS):
            down_out = self.down_proj(intermediate, task_id=task_id)
        else:
            down_out = self.down_proj(intermediate)

        if down_out.dtype != hidden_state.dtype:
            down_out = down_out.to(hidden_state.dtype)
        return down_out

    Phi3MLP.forward = moelora_forward
    print("Patched Phi3MLP.forward() for MMOELoraLinearS projection-local gates")


def verify_replacement(model):
    print("\nVerifying MLP replacement...")
    replaced_count = 0
    total_count = 0

    for layer in model.model.layers:
        if not hasattr(layer, "mlp"):
            continue
        for proj_name in ("gate_up_proj", "down_proj"):
            if not hasattr(layer.mlp, proj_name):
                continue
            total_count += 1
            if isinstance(getattr(layer.mlp, proj_name), MMOELoraLinearS):
                replaced_count += 1

    print(f"  Replaced projections: {replaced_count}/{total_count}")
    if replaced_count == total_count:
        print("  All target projections successfully replaced.")
        return True
    print("  Warning: replacement incomplete.")
    return False


def verify_gate_sharing(model):
    """Verify that every projection has its own gate object."""
    print("\nVerifying projection-local gate configuration...")
    projections = list(iter_moelora_projections(model))
    gate_ids = []
    missing = []

    for layer_idx, proj_name, proj in projections:
        gate = get_moelora_gate(proj)
        if gate is None:
            missing.append(f"{layer_idx}.{proj_name}")
        else:
            gate_ids.append(id(gate))

    unique_gate_count = len(set(gate_ids))
    gate_stats = count_projection_gates(model)
    print(f"  MMOELoRA projections: {len(projections)}")
    print(f"  Gates found:          {len(gate_ids)}")
    print(f"  Unique gate objects:  {unique_gate_count}")
    print(f"  Gate params:          {gate_stats['params']:,}")

    for layer_idx, proj_name, proj in projections[:4]:
        gate = get_moelora_gate(proj)
        status = "independent" if gate is not None and gate_ids.count(id(gate)) == 1 else "shared/missing"
        print(f"    Layer {layer_idx}.{proj_name}: {status}")

    if missing:
        print(f"  Missing task_gate on: {missing[:10]}")
        return False
    if unique_gate_count != len(gate_ids):
        print("  Warning: some projections still share a gate.")
        return False

    print("  Projection-local gate configuration correct.")
    return True


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = 0
    gate_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params += param.numel()
        if is_moelora_gate_parameter_name(name):
            gate_params += param.numel()

    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "gate_a": 0,
        "gate_b": 0,
        "gate_total": gate_params,
        "ratio": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def print_parameter_stats(model):
    stats = count_parameters(model)
    gate_stats = count_projection_gates(model)

    print("\n" + "=" * 60)
    print("Parameter Statistics (Projection-Local Gates):")
    print(f"  Total params:              {stats['total']:>15,}")
    print(f"  Trainable params:          {stats['trainable']:>15,}")
    print(f"  LoRA params:               {stats['lora']:>15,}")
    print(f"  Gate params:               {stats['gate_total']:>15,}")
    print(f"  Gate modules:              {gate_stats['total']:>15,}")
    print(f"  Trainable ratio:           {stats['ratio']:>14.4f}%")
    print("=" * 60)
