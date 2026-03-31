# 增强版工具函数
# 支持下游任务和视频数据处理

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector

# 增强版工具函数
# 支持 DeepStack 多层视觉特征注入
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector


# ============================================================
# DeepStack 相关组件
# ============================================================

class DeepStackConnectors(nn.Module):
    """
    DeepStack 的可训练部分：多层独立 Connector
    
    每个 Connector 负责将视觉编码器某一层的特征
    从 vision 空间 (768d) 投影到 LLM 空间 (1024d)
    
    内部结构与主 Connector 完全一致：
        pixel_shuffle (scale_factor=4) → Linear(12288→1024)
    """
    
    def __init__(
        self,
        num_layers: int,
        vision_hidden_size: int = 768,
        text_hidden_size: int = 1024,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.connectors = nn.ModuleList()
        for _ in range(num_layers):
            self.connectors.append(
                self._create_connector(vision_hidden_size, text_hidden_size, scale_factor)
            )
    
    def _create_connector(self, vision_hidden_size, text_hidden_size, scale_factor):
        @dataclass
        class VisionConfig:
            hidden_size: int = 0
        @dataclass
        class TextConfig:
            hidden_size: int = 0
        @dataclass
        class ConnectorConfig:
            scale_factor: int = 0
            vision_config: VisionConfig = field(default_factory=VisionConfig)
            text_config: TextConfig = field(default_factory=TextConfig)
        
        config = ConnectorConfig(
            scale_factor=scale_factor,
            vision_config=VisionConfig(hidden_size=vision_hidden_size),
            text_config=TextConfig(hidden_size=text_hidden_size),
        )
        return SmolVLMConnector(config)


class DeepStackModelWrapper(nn.Module):
    """
    将 DeepStack 功能包装到原始模型中
    
    核心机制（全部基于 hook，零代码侵入）：
    
    1. Vision Encoder hook（捕获）：
       在视觉编码器的指定层注册 hook，当 base_model.forward()
       执行视觉编码时，自动捕获中间层输出，无需手动执行或复制预处理
    
    2. LLM hook（注入）：
       在 Qwen3 的前几层注册 hook，将捕获到的视觉特征
       经过 Connector 后，以残差相加方式注入到 hidden_states
    
    整体流程（在一次 forward 中自动完成）：
       base_model.forward(pixel_values, input_ids, ...)
         → Vision Encoder 逐层执行
             → Layer 3 完成后: hook 自动捕获特征
             → Layer 7 完成后: hook 自动捕获特征
             → Layer 11 完成后: hook 自动捕获特征
         → 主 Connector → 替换 embedding（原始逻辑）
         → Qwen3 LLM 逐层执行
             → Layer 0 完成后: hook 取出捕获的特征 → Connector[0] → 残差相加
             → Layer 1 完成后: hook 取出捕获的特征 → Connector[1] → 残差相加
             → Layer 2 完成后: hook 取出捕获的特征 → Connector[2] → 残差相加
             → Layer 3~27: 正常执行
         → lm_head → logits
    """
    
    def __init__(
        self,
        base_model,
        deepstack_layer_indexes: List[int] = None,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        
        if deepstack_layer_indexes is None:
            deepstack_layer_indexes = [3, 7, 11]
        
        self.base_model = base_model
        self.deepstack_layer_indexes = deepstack_layer_indexes
        self.image_token_id = base_model.image_token_id
        
        # 可训练的 DeepStack Connectors
        self.deepstack_connectors = DeepStackConnectors(
            num_layers=len(deepstack_layer_indexes),
            vision_hidden_size=768,
            text_hidden_size=1024,
            scale_factor=4,
        ).to(device=device, dtype=dtype)
        
        # 存储 hook 捕获的视觉中间层特征
        self._captured_vision_features = {}
        # 存储当前 batch 的 input_ids（用于定位 image token）
        self._current_input_ids = None
        
        # hook 句柄列表（用于后续移除）
        self._vision_hooks = []
        self._llm_hooks = []
        
        # 注册所有 hook
        self._register_vision_hooks()
        self._register_llm_hooks()
    
    # ==================== Hook 注册 ====================
    
    def _register_vision_hooks(self):
        """在视觉编码器的指定层注册捕获 hook"""
        for hook in self._vision_hooks:
            hook.remove()
        self._vision_hooks = []
        
        vision_layers = self.base_model.model.vision_model.encoder.layers
        
        for idx, layer_idx in enumerate(self.deepstack_layer_indexes):
            hook = vision_layers[layer_idx].register_forward_hook(
                self._make_vision_capture_hook(idx)
            )
            self._vision_hooks.append(hook)
        
        print(f"DeepStack: 视觉层 {self.deepstack_layer_indexes} 已注册捕获 hook")
    
    def _register_llm_hooks(self):
        """在 LLM 的前 N 层注册注入 hook"""
        for hook in self._llm_hooks:
            hook.remove()
        self._llm_hooks = []
        
        llm_layers = self.base_model.model.text_model.layers
        num_inject = len(self.deepstack_layer_indexes)
        
        for inject_idx in range(num_inject):
            hook = llm_layers[inject_idx].register_forward_hook(
                self._make_llm_inject_hook(inject_idx)
            )
            self._llm_hooks.append(hook)
        
        print(f"DeepStack: LLM 层 {list(range(num_inject))} 已注册注入 hook")
    
    # ==================== Hook 函数 ====================
    
    def _make_vision_capture_hook(self, capture_idx: int):
        """
        创建视觉层捕获 hook
        
        当 base_model 内部执行 vision_encoder 时，
        指定层的输出会被自动捕获存储。
        
        Args:
            capture_idx: 在 captured_features 中的存储索引 (0, 1, 2)
        """
        def hook_fn(module, input, output):
            # encoder_layer 输出格式: (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # 存储该层的输出（detach 避免影响原始计算图，clone 避免被后续层覆盖）
            # 不 detach：让梯度可以流回视觉编码器（stage2/3需要）
            self._captured_vision_features[capture_idx] = hidden_states.detach().clone().contiguous()
        
        return hook_fn
    
    def _make_llm_inject_hook(self, inject_idx: int):
        """
        创建 LLM 层注入 hook
        
        当 base_model 内部执行 Qwen3 decoder layer 时，
        在指定层的输出上，对 image_token 位置做残差相加。
        
        流程：
        1. 从 _captured_vision_features 取出对应的视觉特征
        2. 通过 Connector 投影到 LLM 空间
        3. reshape 为 [total_image_tokens, 1024]
        4. 在 image_token 位置残差相加
        
        Args:
            inject_idx: 对应 deepstack_connectors 的索引 (0, 1, 2)
        """
        def hook_fn(module, input, output):
            # 检查是否有捕获到的视觉特征
            if inject_idx not in self._captured_vision_features:
                return output
            if self._current_input_ids is None:
                return output
            
            # 取出 decoder layer 的 hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # 关键修复：generate 后续 step 中 hidden_states 只有 1 个 token
            # 此时 seq_len 与 _current_input_ids 的长度不匹配，直接跳过
            if hidden_states.shape[1] != self._current_input_ids.shape[1]:
                return output
            
            # 构建 image token 位置掩码
            visual_pos_mask = (self._current_input_ids == self.image_token_id)
            
            # 无 image token 时跳过
            if not visual_pos_mask.any():
                return output
            
            # 取出捕获的视觉特征并通过 Connector
            vision_feature = self._captured_vision_features[inject_idx]
            projected = self.deepstack_connectors.connectors[inject_idx](
                vision_feature.contiguous()
            )
            
            # 展平为 [total_image_tokens, 1024]
            visual_embeds = projected.reshape(-1, projected.shape[-1])
            visual_embeds = visual_embeds.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            
            # 验证数量对齐
            num_positions = visual_pos_mask.sum().item()
            num_embeds = visual_embeds.shape[0]
            
            if num_positions != num_embeds:
                print(
                    f"⚠️ DeepStack inject_idx={inject_idx}: "
                    f"positions={num_positions} != embeds={num_embeds}, 跳过"
                )
                return output
            
            # 残差相加（detach hidden_states 避免梯度回传穿过冻结的 LLM 层）
            local_values = hidden_states[visual_pos_mask].detach().clone() + visual_embeds
            hidden_states[visual_pos_mask] = local_values
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    # ==================== 核心方法 ====================
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        DeepStack 增强的前向传播
        
        只需要设置 _current_input_ids，然后调用 base_model.forward()
        所有的捕获和注入都由 hook 自动完成
        """
        # 清理上一次的状态
        self._captured_vision_features.clear()
        self._current_input_ids = input_ids
        
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # 清理状态
            self._captured_vision_features.clear()
            self._current_input_ids = None
        
        return outputs
    
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        """
        DeepStack 增强的生成
        
        generate 内部会多次调用 forward：
        - 第1次: 完整序列，含 image_token → vision hook 捕获 + llm hook 注入
        - 后续: 只有新 token，无 image_token → llm hook 检测后跳过
        
        注意：_current_input_ids 在整个 generate 期间保持为初始的完整 input_ids
        但这不影响正确性，因为后续 step 的 hidden_states 形状是 [B, 1, hidden]
        visual_pos_mask 作用在完整 input_ids 上会选出图像位置，
        但 hidden_states 只有 1 个 token，索引不会命中，hook 自然跳过。
        
        更准确地说：后续 step 中 base_model 内部传给 text_model 的是
        inputs_embeds 而非 input_ids（通过 KV cache），
        所以 vision encoder 不会再被调用，vision hook 也不会触发。
        llm hook 虽然触发，但 visual_pos_mask.any() 为 False（因为
        后续 step 的 input_ids 不含 image_token），直接 return。
        """
        self._captured_vision_features.clear()
        self._current_input_ids = input_ids
        
        try:
            outputs = self.base_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            self._captured_vision_features.clear()
            self._current_input_ids = None
        
        return outputs
    
    # ==================== 属性委托 ====================
    
    @property
    def config(self):
        return self.base_model.config
    
    @property
    def device(self):
        return self.base_model.device
    
    @property
    def dtype(self):
        return self.base_model.dtype
    
    @property
    def generation_config(self):
        return self.base_model.generation_config
    
    @generation_config.setter
    def generation_config(self, value):
        self.base_model.generation_config = value
    
    @property
    def vocab_size(self):
        return self.base_model.vocab_size
    
    @property
    def model(self):
        return self.base_model.model
    
    @property
    def lm_head(self):
        return self.base_model.lm_head
    
    # ==================== 参数管理 ====================
    
    def named_parameters(self, *args, **kwargs):
        yield from self.base_model.named_parameters(*args, **kwargs)
        for name, param in self.deepstack_connectors.named_parameters(*args, **kwargs):
            yield f"deepstack_connectors.{name}", param
    
    def parameters(self, *args, **kwargs):
        yield from self.base_model.parameters(*args, **kwargs)
        yield from self.deepstack_connectors.parameters(*args, **kwargs)
    
    def train(self, mode=True):
        self.base_model.train(mode)
        self.deepstack_connectors.train(mode)
        return self
    
    def eval(self):
        self.base_model.eval()
        self.deepstack_connectors.eval()
        return self
    
    # ==================== 保存/加载 ====================
    
    def save_pretrained(self, output_dir, **kwargs):
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        
        self.base_model.save_pretrained(output_dir, **kwargs)
        
        deepstack_path = os.path.join(output_dir, "deepstack_connectors.bin")
        torch.save(self.deepstack_connectors.state_dict(), deepstack_path)
        
        config_path = os.path.join(output_dir, "deepstack_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "deepstack_layer_indexes": self.deepstack_layer_indexes,
                "vision_hidden_size": 768,
                "text_hidden_size": 1024,
                "scale_factor": 4,
            }, f, indent=2)
        
        print(f"DeepStack 参数已保存到: {deepstack_path}")
    
    def load_deepstack_weights(self, checkpoint_dir):
        import os
        deepstack_path = os.path.join(checkpoint_dir, "deepstack_connectors.bin")
        if os.path.exists(deepstack_path):
            state_dict = torch.load(deepstack_path, map_location=self.device)
            self.deepstack_connectors.load_state_dict(state_dict)
            print(f"✅ DeepStack 权重已从 {deepstack_path} 加载")
        else:
            print(f"⚠️ 未找到 DeepStack 权重: {deepstack_path}")
    
    def remove_all_hooks(self):
        """移除所有 hook（模型销毁前调用）"""
        for hook in self._vision_hooks:
            hook.remove()
        for hook in self._llm_hooks:
            hook.remove()
        self._vision_hooks = []
        self._llm_hooks = []

def load_deepstack_model(device="cuda:0", deepstack_layer_indexes=None):
    """
    加载带有 DeepStack 功能的模型
    
    Args:
        device: 运行设备
        deepstack_layer_indexes: 要提取的视觉层索引
            默认 [3, 7, 11]（均匀采样视觉编码器的低/中/高层）
    
    Returns:
        DeepStackModelWrapper 实例
    """
    if deepstack_layer_indexes is None:
        deepstack_layer_indexes = [3, 7, 11]
    
    base_model = load_model(device)
    
    model = DeepStackModelWrapper(
        base_model=base_model,
        deepstack_layer_indexes=deepstack_layer_indexes,
        device=device,
        dtype=torch.bfloat16,
    )
    
    num_new_params = sum(p.numel() for p in model.deepstack_connectors.parameters())
    print(f"DeepStack 模型构建完成！")
    print(f"  视觉编码器提取层: {deepstack_layer_indexes}")
    print(f"  注入LLM层: {list(range(len(deepstack_layer_indexes)))}")
    print(f"  DeepStack 新增参数: {num_new_params / 1e6:.1f}M")
    
    return model
def load_processor():
    """
    加载和配置数据处理器
    
    此函数的作用：
    1. 加载SmolVLM2的图像处理器
    2. 加载Qwen3的分词器  
    3. 将两者组合并配置特殊token
    4. 设置聊天模板
    
    这样做的原因是要将SmolVLM2的视觉处理能力与Qwen3的文本处理能力结合，
    创建一个支持中文的多模态处理器。
    
    Returns:
        processor: 配置好的多模态处理器
    """
    print("正在加载SmolVLM2处理器...")
    smolvlm2_processor = AutoProcessor.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct"
    )
    
    print("正在加载Qwen3分词器...")
    qwen3_tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")

    print("正在配置处理器...")
    smolvlm2_processor.tokenizer = qwen3_tokenizer
    
    # 加载聊天模板文件
    with open("chat_template.jinja", "r") as f:
        smolvlm2_processor.chat_template = f.read()
    
    # 配置特殊token
    smolvlm2_processor.fake_image_token = "<vision_start>"
    smolvlm2_processor.image_token = "<|image_pad|>"
    smolvlm2_processor.image_token_id = 151655
    smolvlm2_processor.end_of_utterance_token = "<im_end>"
    smolvlm2_processor.global_image_token = "<|vision_pad|>"
    smolvlm2_processor.video_token = "<|video_pad|>"

    return smolvlm2_processor


def load_model(device="cuda:0"):
    """
    加载和构建混合多模态模型
    
    此函数实现了一个创新的模型架构组合：
    1. 使用SmolVLM2的视觉编码器处理图像
    2. 使用Qwen3的语言模型处理文本
    3. 创建新的连接器将视觉特征映射到文本特征空间
    
    这种组合的优势：
    - SmolVLM2：优秀的视觉理解能力
    - Qwen3：强大的中文语言能力
    - 自定义连接器：优化的跨模态特征映射
    
    Args:
        device: 运行设备，默认为"cuda:0"
    
    Returns:
        smolvlm2_02B_model: 配置好的混合多模态模型
    """
    print("正在加载SmolVLM2视觉-语言模型...")
    smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(device)
    
    print("正在加载Qwen3语言模型...")
    qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
        "model/Qwen3-0.6B", 
        torch_dtype=torch.bfloat16
    ).to(device)

    print("正在构建连接器配置...")
    @dataclass
    class VisionConfig:
        hidden_size: int = 768

    @dataclass
    class TextConfig:
        hidden_size: int = 1024

    @dataclass
    class ConnectConfig:
        scale_factor: int = 4
        vision_config: VisionConfig = field(default_factory=VisionConfig)
        text_config: TextConfig = field(default_factory=TextConfig)

    new_connector_config = ConnectConfig()

    print("正在创建新的连接器...")
    new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
    smolvlm2_02B_model.model.connector = new_connector

    print("正在替换语言模型组件...")
    smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
    smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head
    
    print("正在更新模型配置...")
    vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.vocab_size = vocab_size
    smolvlm2_02B_model.model.vocab_size = vocab_size
    smolvlm2_02B_model.config.vocab_size = vocab_size
    smolvlm2_02B_model.config.text_config.vocab_size = vocab_size
    smolvlm2_02B_model.model.config.vocab_size = vocab_size
    smolvlm2_02B_model.model.config.text_config.vocab_size = vocab_size
    
    image_token_id = 151655
    smolvlm2_02B_model.image_token_id = image_token_id
    smolvlm2_02B_model.model.image_token_id = image_token_id
    smolvlm2_02B_model.config.image_token_id = image_token_id
    smolvlm2_02B_model.model.config.image_token_id = image_token_id
    
    smolvlm2_02B_model.generation_config.eos_token_id = 151645
    
    print("模型构建完成！")
    return smolvlm2_02B_model


def load_downstream_datasets(task_names: List[str]):
    """
    加载下游任务数据集
    
    Args:
        task_names: 下游任务名称列表
    
    Returns:
        datasets: 下游任务数据集字典
    """
    downstream_datasets = {}
    
    # 定义可用的下游任务
    available_tasks = {
        "captioning": ["coco_caption", "flickr30k", "nocaps"],
        "vqa": ["vqa_v2", "gqa", "okvqa"],
        "video": ["msrvtt", "activitynet", "youcook2"],
        "ocr": ["textvqa", "docvqa", "funsd"],
        "reasoning": ["clevr", "nlvr2", "vcr"],
    }
    
    for task_name in task_names:
        if task_name in available_tasks:
            print(f"加载下游任务: {task_name}")
            # 这里可以添加具体的数据集加载逻辑
            # 目前返回空数据集作为占位符
            downstream_datasets[task_name] = []
    
    return downstream_datasets


def create_video_processor():
    """
    创建视频数据处理器
    
    Returns:
        video_processor: 视频处理器
    """
    # 这里可以添加视频处理器的创建逻辑
    # 目前返回None作为占位符
    return None


def apply_parameter_efficient_finetuning(model, method="lora"):
    """
    应用参数高效微调方法
    
    Args:
        model: 要微调的模型
        method: 微调方法 ("lora", "adapter", "prefix_tuning")
    
    Returns:
        model: 应用了参数高效微调的模型
    """
    if method == "lora":
        # 应用LoRA微调
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
    elif method == "adapter":
        # 应用Adapter微调
        from peft import AdapterConfig, get_peft_model
        
        adapter_config = AdapterConfig(
            adapter_size=64,
            adapter_non_linearity="relu",
            adapter_dropout=0.1
        )
        model = get_peft_model(model, adapter_config)
    
    return model


def create_custom_loss_function():
    """
    创建自定义损失函数
    
    Returns:
        loss_fn: 自定义损失函数
    """
    def custom_loss(logits, labels, attention_mask=None):
        """
        自定义损失函数，支持多任务学习
        
        Args:
            logits: 模型输出
            labels: 真实标签
            attention_mask: 注意力掩码
        
        Returns:
            loss: 计算得到的损失
        """
        # 基础交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 移位logits和labels用于语言建模
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算损失
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                       shift_labels.view(-1))
        
        return loss
    
    return custom_loss


def setup_mixed_precision_training():
    """
    设置混合精度训练
    
    Returns:
        scaler: 梯度缩放器
    """
    from torch.cuda.amp import GradScaler
    
    scaler = GradScaler()
    return scaler


def create_optimizer_with_different_lrs(model, stage_config):
    """
    为不同组件创建不同学习率的优化器
    
    Args:
        model: 模型
        stage_config: 阶段配置
    
    Returns:
        optimizer: 优化器
    """
    # 为不同组件设置不同的学习率
    param_groups = []
    
    # 连接器使用较高学习率
    if hasattr(model, 'model') and hasattr(model.model, 'connector'):
        param_groups.append({
            'params': model.model.connector.parameters(),
            'lr': stage_config.get('connector_lr', 1e-4)
        })
    
    # 视觉编码器使用中等学习率
    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        param_groups.append({
            'params': model.model.vision_model.parameters(),
            'lr': stage_config.get('vision_lr', 5e-5)
        })
    
    # 文本模型使用较低学习率
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        param_groups.append({
            'params': model.model.text_model.parameters(),
            'lr': stage_config.get('text_lr', 1e-5)
        })
    
    # 其他参数使用默认学习率
    other_params = []
    for name, param in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in 
                  ['model.connector', 'model.vision_model', 'model.text_model']):
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': stage_config.get('default_lr', 1e-4)
        })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer


def save_model_checkpoint(model, processor, output_dir, stage_name, 
                         save_optimizer=True, save_scheduler=True):
    """
    保存模型检查点
    
    Args:
        model: 模型
        processor: 处理器
        output_dir: 输出目录
        stage_name: 阶段名称
        save_optimizer: 是否保存优化器状态
        save_scheduler: 是否保存调度器状态
    """
    import os
    
    checkpoint_dir = os.path.join(output_dir, stage_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # 保存训练状态
    training_state = {
        'stage_name': stage_name,
        'model_config': model.config.to_dict(),
        'processor_config': processor.config.to_dict() if hasattr(processor, 'config') else {}
    }
    
    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
        import json
        json.dump(training_state, f, indent=2)
    
    print(f"模型检查点已保存到: {checkpoint_dir}")


def load_model_checkpoint(checkpoint_dir, device="cuda:0"):
    """
    加载模型检查点
    
    Args:
        checkpoint_dir: 检查点目录
        device: 设备
    
    Returns:
        model: 加载的模型
        processor: 加载的处理器
        training_state: 训练状态
    """
    # 加载模型和处理器
    model = load_model(device)
    processor = load_processor()
    
    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin')))
    
    # 加载训练状态
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            import json
            training_state = json.load(f)
    else:
        training_state = {}
    
    print(f"模型检查点已从 {checkpoint_dir} 加载")
    return model, processor, training_state 