#!/usr/bin/env python3
"""
简化的多模态推理代码，支持对比测试：
1. Qwen3-SmVL（训练后的混合模型）
2. Qwen3.5-0.8B（纯文本基线）
3. SmolVLM2-256M（原始多模态基线）
"""

import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from utils import load_model, load_processor


def load_trained_model(checkpoint_path, device="cuda"):
    """
    加载训练后的模型
    
    Args:
        checkpoint_path: 训练后模型的路径
        device: 运行设备
        
    Returns:
        model, processor
    """
    print(f"正在加载训练后的模型: {checkpoint_path}")
    
    # 使用原始的模型构建方式
    model = load_model(device)
    processor = load_processor()
    
    # 加载训练后的权重（优先pytorch_model.bin）
    if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        print("正在加载pytorch权重...")
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
    elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        print("正在加载safetensors权重...")
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
    else:
        print("⚠️  未找到权重文件，使用原始模型")
    
    model.eval()
    return model, processor


def inference(model, processor, image_path, prompt, max_tokens=512, device="cuda"):
    """
    简单的推理函数
    
    Args:
        model: 加载的模型
        processor: 处理器
        image_path: 图像路径
        prompt: 文本提示
        max_tokens: 最大token数
        device: 设备
        
    Returns:
        生成的文本
    """
    # 加载图像
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    messages = [
        {
            "role": "system",
            "content": "使用中文回答所有问题。",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = inputs.to(device)
    
    # 确保输入数据类型与模型权重匹配（bfloat16）
    for key in inputs:
        if key == 'pixel_values' and inputs[key] is not None:
            inputs[key] = inputs[key].to(torch.bfloat16)
        elif key == 'input_ids' and inputs[key] is not None:
            inputs[key] = inputs[key].to(device)
        elif key == 'attention_mask' and inputs[key] is not None:
            inputs[key] = inputs[key].to(device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True
        )
    
    # 解码输出
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def load_qwen3_5(device="cuda"):
    """加载 Qwen3.5-0.8B 原生多模态模型"""
    model_path = "./model/Qwen3.5-0.8B"
    print(f"正在加载 Qwen3.5-0.8B: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, processor


def inference_qwen3_5(model, processor, image_path, prompt, max_tokens=512, device="cuda"):
    """Qwen3.5-0.8B 多模态推理"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

    messages = [
        {"role": "system", "content": "使用中文回答所有问题。"},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    for key in inputs:
        if key == 'pixel_values' and inputs[key] is not None:
            inputs[key] = inputs[key].to(torch.bfloat16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def load_smolvlm2(device="cuda"):
    """加载原始 SmolVLM2-256M-Video-Instruct 模型"""
    model_path = "./model/SmolVLM2-256M-Video-Instruct"
    print(f"正在加载 SmolVLM2-256M: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, _attn_implementation="eager"
    ).to(device).eval()
    return model, processor


def inference_smolvlm2(model, processor, image_path, prompt, max_tokens=512, device="cuda"):
    """SmolVLM2-256M 多模态推理"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    for key in inputs:
        if key == 'pixel_values' and inputs[key] is not None:
            inputs[key] = inputs[key].to(torch.bfloat16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def run_test(name, infer_fn, prompts):
    """运行一组推理测试并打印结果"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for i, (prompt, kwargs) in enumerate(prompts, 1):
        print(f"\n{i}. 提示: {prompt}")
        print("-" * 50)
        try:
            response = infer_fn(prompt, **kwargs)
            print(f"回复: {response}")
        except Exception as e:
            print(f"推理失败: {e}")


def main():
    """主函数：对比三个模型的推理结果"""
    trained_model_path = "./model/staged_training_test/stage2"
    image_path = "./resource/dog.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts_vision = [
        "请描述这张图片。",
        "图片中有什么东西？",
        "图中的数量有多少？"
    ]

    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return

    # ========== 1. Qwen3-SmVL（训练后的混合模型） ==========
    if os.path.exists(trained_model_path):
        try:
            model, processor = load_trained_model(trained_model_path, device)
            run_test(
                "Qwen3-SmVL（训练后模型）",
                lambda p, **kw: inference(model, processor, image_path, p, device=device),
                [(p, {}) for p in prompts_vision],
            )
            del model, processor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Qwen3-SmVL 加载失败: {e}")
    else:
        print(f"\n跳过 Qwen3-SmVL：路径不存在 {trained_model_path}")

    # ========== 2. Qwen3.5-0.8B（原生多模态基线） ==========
    try:
        model_q, processor_q = load_qwen3_5(device)
        run_test(
            "Qwen3.5-0.8B（原生多模态基线）",
            lambda p, **kw: inference_qwen3_5(model_q, processor_q, image_path, p, device=device),
            [(p, {}) for p in prompts_vision],
        )
        del model_q, processor_q
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Qwen3.5-0.8B 加载失败: {e}")

    # ========== 3. SmolVLM2-256M（原始多模态基线） ==========
    try:
        model_s, processor_s = load_smolvlm2(device)
        run_test(
            "SmolVLM2-256M（原始多模态基线）",
            lambda p, **kw: inference_smolvlm2(model_s, processor_s, image_path, p, device=device),
            [(p, {}) for p in prompts_vision],
        )
        del model_s, processor_s
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"SmolVLM2-256M 加载失败: {e}")

    print(f"\n{'='*60}")
    print("全部对比测试完成！")


if __name__ == "__main__":
    main()
