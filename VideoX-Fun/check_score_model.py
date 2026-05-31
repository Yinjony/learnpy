import torch
from safetensors.torch import load_file

# ====== 第1步：检查score_modules里的权重 ======
score_ckpt = load_file("/root/data1/yizeli/VideoX-Fun/scripts/wan2.1/output_dir_cpo_stage1_1.3b/score_modules-10000.safetensors")

print(f"Score modules keys: {len(score_ckpt)}")
print(f"Total params: {sum(v.numel() for v in score_ckpt.values()) / 1e6:.1f}M")
print(f"File size: {sum(v.numel() * v.element_size() for v in score_ckpt.values()) / 1e9:.2f}GB")
print()

# 检查几个关键层的值
for key in sorted(score_ckpt.keys())[:20]:
    v = score_ckpt[key]
    print(f"{key}: shape={list(v.shape)}, mean={v.float().mean():.6f}, std={v.float().std():.6f}, min={v.float().min():.6f}, max={v.float().max():.6f}")

print("\n... (showing first 20 keys)")
print(f"\nTotal keys: {len(score_ckpt)}")

# ====== 第2步：检查out_proj是否还接近零 ======
print("\n====== out_proj weights (should be near zero after 50 steps) ======")
for key in sorted(score_ckpt.keys()):
    if 'out_proj.weight' in key:
        v = score_ckpt[key]
        print(f"{key}: std={v.float().std():.8f}, max_abs={v.float().abs().max():.8f}")
        break  # 只看第一个block

# ====== 第3步：检查是否所有值都是零（没训上） ======
all_zero_count = 0
for key, v in score_ckpt.items():
    if v.float().abs().max() == 0:
        all_zero_count += 1
print(f"\nAll-zero tensors: {all_zero_count} / {len(score_ckpt)}")

# ====== 第4步：检查quality和motion是否不同（正交性初步验证）======
q_params = []
m_params = []
for key, v in score_ckpt.items():
    if 'quality_' in key and 'weight' in key:
        q_params.append(v.float().flatten())
    if 'motion_' in key and 'weight' in key:
        m_params.append(v.float().flatten())

if q_params and m_params:
    q_cat = torch.cat(q_params)
    m_cat = torch.cat(m_params)
    min_len = min(len(q_cat), len(m_cat))
    cos_sim = torch.nn.functional.cosine_similarity(q_cat[:min_len].unsqueeze(0), m_cat[:min_len].unsqueeze(0))
    print(f"\nQuality vs Motion cosine similarity: {cos_sim.item():.6f}")
    print("(should be near 0 if orthogonal, near 1 if collapsed)")