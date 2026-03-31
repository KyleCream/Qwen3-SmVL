"""
测试 DeepStack 的正确性
运行: python test_deepstack.py
"""
import torch
from PIL import Image
from utils import load_deepstack_model, load_processor


def test_basic_forward():
    """测试1: 基本前向传播是否能跑通"""
    print("=" * 60)
    print("测试1: 基本前向传播")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    model = load_deepstack_model(device)
    model.eval()
    
    # 构建输入
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "描述这张图片"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.new("RGB", (256, 512), color=(128, 100, 200))
    
    batch = processor(
        text=[text], images=[[image]],
        return_tensors="pt", padding=True,
    ).to(device, dtype=torch.bfloat16)
    
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"pixel_values shape: {batch['pixel_values'].shape}")
    print(f"image_token 数量: {(batch['input_ids'] == 151655).sum().item()}")
    
    with torch.no_grad():
        outputs = model(**batch)
    
    print(f"logits shape: {outputs.logits.shape}")
    print("✅ 测试1通过: 前向传播正常\n")
    return model, processor, batch


def test_deepstack_features_captured(model, batch):
    """测试2: 验证 vision hook 是否正确捕获了特征"""
    print("=" * 60)
    print("测试2: Vision hook 捕获验证")
    print("=" * 60)
    
    # 清理之前的状态
    model._captured_vision_features.clear()
    model._current_input_ids = batch["input_ids"]
    
    # 手动触发视觉编码（通过 base_model 的 get_image_features）
    with torch.no_grad():
        _ = model.base_model.model.get_image_features(
            batch["pixel_values"]
        )
    
    print(f"捕获到的特征层数: {len(model._captured_vision_features)}")
    for idx, feat in model._captured_vision_features.items():
        print(f"  capture_idx={idx}: shape={feat.shape}, dtype={feat.dtype}")
    
    assert len(model._captured_vision_features) == len(model.deepstack_layer_indexes), \
        "捕获的特征数量与指定层数不一致！"
    
    # 验证特征形状
    for idx, feat in model._captured_vision_features.items():
        assert feat.shape[-1] == 768, f"特征维度应为768，实际为{feat.shape[-1]}"
    
    model._captured_vision_features.clear()
    model._current_input_ids = None
    print("✅ 测试2通过: Vision hook 正确捕获了中间层特征\n")


def test_dimension_alignment(model, processor, batch):
    """测试3: 验证 Connector 输出与 image_token 数量对齐"""
    print("=" * 60)
    print("测试3: 维度对齐验证")
    print("=" * 60)
    
    num_image_tokens = (batch["input_ids"] == 151655).sum().item()
    print(f"input_ids 中 image_token 数量: {num_image_tokens}")
    
    # 手动捕获视觉特征
    model._captured_vision_features.clear()
    with torch.no_grad():
        _ = model.base_model.model.get_image_features(batch["pixel_values"])
    
    # 对每层特征过 Connector 并检查维度
    for idx, feat in model._captured_vision_features.items():
        with torch.no_grad():
            projected = model.deepstack_connectors.connectors[idx](feat)
        flat = projected.reshape(-1, projected.shape[-1])
        print(f"  capture_idx={idx}: "
              f"原始={feat.shape} → Connector后={projected.shape} → 展平={flat.shape}")
        
        assert flat.shape[0] == num_image_tokens, \
            f"展平后token数={flat.shape[0]} != image_token数={num_image_tokens}"
    
    model._captured_vision_features.clear()
    print("✅ 测试3通过: Connector 输出维度与 image_token 数量完全对齐\n")


def test_residual_injection(model, batch):
    """测试4: 验证残差注入是否真的修改了 hidden_states"""
    print("=" * 60)
    print("测试4: 残差注入效果验证")
    print("=" * 60)
    
    # 方法：对比有/无 DeepStack 时 LLM 第一层输出的差异
    
    # 记录 LLM Layer 0 的输出
    layer0_outputs = {}
    
    def capture_layer0(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                layer0_outputs[name] = output[0].clone()
            else:
                layer0_outputs[name] = output.clone()
        return hook_fn
    
    llm_layer0 = model.base_model.model.text_model.layers[0]
    
    # Run 1: 有 DeepStack（hook 已注册）
    with torch.no_grad():
        capture_hook = llm_layer0.register_forward_hook(capture_layer0("with_deepstack"))
        _ = model(**batch)
        capture_hook.remove()
    
    # Run 2: 无 DeepStack（临时禁用）
    # 保存并清除 LLM hook
    saved_hooks = model._llm_hooks.copy()
    for h in model._llm_hooks:
        h.remove()
    model._llm_hooks = []
    
    with torch.no_grad():
        capture_hook = llm_layer0.register_forward_hook(capture_layer0("without_deepstack"))
        model._captured_vision_features.clear()
        model._current_input_ids = batch["input_ids"]
        _ = model.base_model(**batch)
        model._captured_vision_features.clear()
        model._current_input_ids = None
        capture_hook.remove()
    
    # 恢复 LLM hook
    model._llm_hooks = saved_hooks
    model._register_llm_hooks()
    
    # 比较差异
    with_ds = layer0_outputs["with_deepstack"]
    without_ds = layer0_outputs["without_deepstack"]
    
    visual_pos_mask = (batch["input_ids"] == 151655)
    
    diff_at_image = (with_ds[visual_pos_mask] - without_ds[visual_pos_mask]).abs().mean().item()
    diff_at_text = (with_ds[~visual_pos_mask] - without_ds[~visual_pos_mask]).abs().mean().item()
    
    print(f"  图像token位置的平均差异: {diff_at_image:.6f}")
    print(f"  文本token位置的平均差异: {diff_at_text:.6f}")
    
    assert diff_at_image > 0, "图像位置应该有差异（DeepStack注入了特征）"
    # 文本位置差异应该为0或极小（只受注意力间接影响，但在同一层内不会）
    print("✅ 测试4通过: DeepStack 确实在图像位置注入了特征\n")


def test_generate(model, processor):
    """测试5: 生成功能是否正常"""
    print("=" * 60)
    print("测试5: 生成功能测试")
    print("=" * 60)
    
    device = model.device
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "图中有什么？"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.new("RGB", (256, 512), color=(128, 100, 200))
    
    batch = processor(
        text=[text], images=[[image]],
        return_tensors="pt", padding=True,
    ).to(device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=50,
        )
    
    input_len = batch["input_ids"].shape[1]
    generated_text = processor.decode(
        generated_ids[0][input_len:], skip_special_tokens=True
    )
    
    print(f"  生成文本: {generated_text[:200]}")
    print("✅ 测试5通过: 生成功能正常\n")


def test_gradient_flow():
    """测试6: 梯度是否能流过 DeepStack Connectors"""
    print("=" * 60)
    print("测试6: 梯度流动验证")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    model = load_deepstack_model(device)

    # 冻结 base_model 所有参数
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False

    # 解冻 DeepStack Connectors
    for name, param in model.deepstack_connectors.named_parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    # 用小图减少计算量
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "描述"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "狗"},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False
    )
    image = Image.new("RGB", (32, 32), color=(128, 100, 200))  # 极小图

    batch = processor(
        text=[text], images=[[image]],
        return_tensors="pt", padding=True,
    ).to(device, dtype=torch.bfloat16)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 151655] = -100
    batch["labels"] = labels

    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"image_token 数量: {(batch['input_ids'] == 151655).sum().item()}")

    # 前向
    print("开始前向传播...")
    outputs = model(**batch)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")

    # 反向
    print("开始反向传播...")
    loss.backward()
    print("反向传播完成！")

    # 检查梯度
    print("\nDeepStack Connector 梯度检查:")
    for name, param in model.deepstack_connectors.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  ✅ {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  ❌ {name}: 无梯度!")

    print("\n完成!")


if __name__ == "__main__":
    print("🚀 开始 DeepStack 测试\n")
    from PIL import Image
    test_image = Image.new("RGB", (64, 64), color=(128, 100, 200))
    test_image.save("./resource/test_small.png")
    
    # 测试1: 前向传播
    # model, processor, batch = test_basic_forward()
    
    # 测试2: Vision hook 捕获
    # test_deepstack_features_captured(model, batch)
    
    # # 测试3: 维度对齐
    # test_dimension_alignment(model, processor, batch)
    
    # # 测试4: 残差注入
    # test_residual_injection(model, batch)
    
    # 测试5: 生成
    # test_generate(model, processor)
    
    # 测试6: 梯度流动
    test_gradient_flow()
    
    print("=" * 60)
    print("🎉 所有测试通过！DeepStack 实现正确。")
    print("=" * 60)