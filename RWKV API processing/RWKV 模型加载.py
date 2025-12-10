"""
RWKV 原生 .pth/.pkl 文件加载方案
包装为支持 hidden_states 输出的 PyTorch 模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import os


class RWKVNativeWrapper(nn.Module):
    """
    将原生 RWKV 模型包装为兼容 HuggingFace 接口的模型
    支持输出中间层状态
    """

    def __init__(self, model_path: str, strategy: str = 'cpu fp32'):
        super().__init__()

        # 检查文件格式
        if not (model_path.endswith('.pth') or model_path.endswith('.pkl')):
            raise ValueError("模型文件必须是 .pth 或 .pkl 格式")

        # 方法 1: 使用官方 RWKV pip 包
        try:
            from rwkv.model import RWKV
            self.rwkv = RWKV(model=model_path, strategy=strategy)
            self.use_official = True
            print(f"✓ 使用官方 RWKV 库加载: {model_path}")
        except ImportError:
            # 方法 2: 直接加载权重字典
            print("官方 RWKV 库未安装,尝试直接加载权重...")
            self.weights = torch.load(model_path, map_location='cpu')
            self.use_official = False
            self._init_from_weights()

        # 提取模型配置
        self._extract_config()

        # 用于存储中间状态
        self.hidden_states_cache = []
        self.return_hidden = False

    def _extract_config(self):
        """从权重中提取模型配置"""
        if self.use_official:
            # 从 RWKV 对象提取
            self.n_layer = self.rwkv.args.n_layer
            self.n_embd = self.rwkv.args.n_embd
            self.vocab_size = self.rwkv.args.vocab_size
        else:
            # 从权重字典推断
            self.n_layer = max([
                int(k.split('.')[1])
                for k in self.weights.keys()
                if k.startswith('blocks.')
            ]) + 1

            # 查找嵌入维度
            emb_key = 'emb.weight'
            if emb_key in self.weights:
                self.vocab_size, self.n_embd = self.weights[emb_key].shape
            else:
                raise ValueError("无法从权重中提取配置")

        print(f"模型配置: {self.n_layer} 层, {self.n_embd} 维度, {self.vocab_size} 词汇量")

    def _init_from_weights(self):
        """直接从权重字典初始化 (不使用官方库)"""
        # 这需要手动实现 RWKV 的前向传播
        # 这里提供简化版框架
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.emb.weight.data = self.weights['emb.weight']

        # 初始化各层 (需要根据 RWKV 架构实现)
        self.blocks = nn.ModuleList([
            self._build_rwkv_block(i) for i in range(self.n_layer)
        ])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # 加载权重
        self._load_weights()

    def _build_rwkv_block(self, layer_idx: int):
        """构建单个 RWKV block (简化版)"""

        # 实际需要实现完整的 Time-mixing 和 Channel-mixing
        class RWKVBlock(nn.Module):
            def __init__(self, n_embd):
                super().__init__()
                self.ln1 = nn.LayerNorm(n_embd)
                self.ln2 = nn.LayerNorm(n_embd)
                # ... 其他层

            def forward(self, x, state):
                # 简化的前向传播
                return x, state

        return RWKVBlock(self.n_embd)

    def _load_weights(self):
        """从权重字典加载到 nn.Module"""
        # 映射权重键到模块
        for name, param in self.named_parameters():
            weight_key = name  # 可能需要键名转换
            if weight_key in self.weights:
                param.data = self.weights[weight_key]

    def forward(self,
                input_ids: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                return_hidden_states: bool = True) -> dict:
        """
        前向传播,兼容 HuggingFace 接口

        Args:
            input_ids: [batch_size, seq_len]
            state: RWKV 的循环状态
            return_hidden_states: 是否返回中间层状态

        Returns:
            {
                'logits': [batch_size, seq_len, vocab_size],
                'hidden_states': List[Tensor] 或 None,
                'state': 最终状态
            }
        """
        self.return_hidden = return_hidden_states
        self.hidden_states_cache = []

        batch_size, seq_len = input_ids.shape

        if self.use_official:
            # 使用官方库 (需要逐 token 处理)
            return self._forward_official(input_ids, state)
        else:
            # 使用自定义实现
            return self._forward_custom(input_ids, state)

    def _forward_official(self, input_ids: torch.Tensor, state) -> dict:
        """使用官方 RWKV 库的前向传播"""
        batch_size, seq_len = input_ids.shape

        if batch_size > 1:
            raise NotImplementedError("官方 RWKV 库不支持 batch_size > 1")

        all_logits = []

        for i in range(seq_len):
            token = int(input_ids[0, i].item())

            # 调用官方接口
            logits, state = self.rwkv.forward(token, state)
            all_logits.append(logits)

            # 保存中间状态 (state 是一个列表,每层一个元素)
            if self.return_hidden and state is not None:
                # RWKV state 格式: [n_layer, 4, n_embd]
                # 转换为类似 Transformer hidden_states
                layer_states = []
                for layer_state in state:
                    # 提取每层的隐藏向量 (取状态的第一个通道作为代表)
                    if isinstance(layer_state, np.ndarray):
                        layer_hidden = torch.from_numpy(layer_state[0])  # [n_embd]
                    else:
                        layer_hidden = layer_state[0]
                    layer_states.append(layer_hidden.unsqueeze(0).unsqueeze(0))  # [1, 1, n_embd]

                if i == 0:  # 只存储一次 (或每个位置都存)
                    self.hidden_states_cache = layer_states

        # 拼接 logits
        logits_tensor = torch.tensor(np.array(all_logits)).unsqueeze(0)  # [1, seq_len, vocab_size]

        return {
            'logits': logits_tensor,
            'hidden_states': self.hidden_states_cache if self.return_hidden else None,
            'state': state
        }

    def _forward_custom(self, input_ids: torch.Tensor, state) -> dict:
        """自定义实现的前向传播"""
        # 嵌入
        x = self.emb(input_ids)  # [batch, seq_len, n_embd]

        hidden_states = []

        # 逐层传播
        for i, block in enumerate(self.blocks):
            x, state = block(x, state)

            if self.return_hidden:
                hidden_states.append(x.clone())

        # 输出层
        x = self.ln_out(x)
        logits = self.head(x)

        return {
            'logits': logits,
            'hidden_states': hidden_states if self.return_hidden else None,
            'state': state
        }

    def get_input_embeddings(self):
        """返回嵌入层 (兼容 HuggingFace)"""
        if self.use_official:
            # 从官方模型提取
            emb_weight = self.rwkv.w['emb.weight']
            return nn.Embedding.from_pretrained(torch.from_numpy(emb_weight))
        else:
            return self.emb


class RWKVTokenizerWrapper:
    """包装 RWKV 分词器以兼容 HuggingFace 接口"""

    def __init__(self, vocab_path: str = None):
        try:
            # 方法 1: 使用官方分词器
            from rwkv.utils import TOKENIZER
            if vocab_path and os.path.exists(vocab_path):
                self.tokenizer = TOKENIZER(vocab_path)
            else:
                # 使用世界模型分词器
                self.tokenizer = TOKENIZER("rwkv_vocab_v20230424")
            self.use_official = True
            print("✓ 使用官方 RWKV 分词器")
        except:
            # 方法 2: 使用 HuggingFace 分词器
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "RWKV/rwkv-4-world-3b",
                trust_remote_code=True
            )
            self.use_official = False
            print("✓ 使用 HuggingFace 分词器")

    def __call__(self, text: str, return_tensors: str = 'pt', **kwargs):
        """分词接口"""
        if self.use_official:
            tokens = self.tokenizer.encode(text)
            if return_tensors == 'pt':
                return {'input_ids': torch.tensor([tokens])}
            return {'input_ids': tokens}
        else:
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)

    def decode(self, token_ids, **kwargs):
        """解码接口"""
        if self.use_official:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.squeeze().tolist()
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids, **kwargs)

    def tokenize(self, text: str):
        """返回 token 字符串列表"""
        if self.use_official:
            tokens = self.tokenizer.encode(text)
            return [self.tokenizer.decode([t]) for t in tokens]
        else:
            return self.tokenizer.tokenize(text)

    @property
    def pad_token_id(self):
        return 0

    @property
    def mask_token_id(self):
        return 0


# ============ 使用示例 ============

def load_rwkv_native(model_path: str, vocab_path: str = None):
    """
    加载原生 RWKV 模型的便捷函数

    Args:
        model_path: .pth 或 .pkl 文件路径
        vocab_path: 词汇表文件路径 (可选)

    Returns:
        model, tokenizer
    """
    print(f"加载 RWKV 模型: {model_path}")

    # 加载模型
    model = RWKVNativeWrapper(model_path, strategy='cpu fp32')

    # 加载分词器
    tokenizer = RWKVTokenizerWrapper(vocab_path)

    return model, tokenizer


if __name__ == "__main__":
    # 使用示例
    model_path = "RWKV-4-World-3B-v1-20230619-ctx4096.pth"

    # 加载模型
    model, tokenizer = load_rwkv_native(model_path)

    # 测试前向传播
    text = "Hello world"
    inputs = tokenizer(text, return_tensors='pt')

    outputs = model(inputs['input_ids'], return_hidden_states=True)

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states: {len(outputs['hidden_states'])} layers")

    # 集成到分析管道
    from rwkv_analysis_pipeline import RWKVAnalysisPipeline, AnalysisConfig

    config = AnalysisConfig(
        model_path=model_path,
        num_layers=model.n_layer
    )


    # 修改管道的加载函数
    class CustomPipeline(RWKVAnalysisPipeline):
        def _load_rwkv_model(self):
            model, _ = load_rwkv_native(self.config.model_path)
            return model

        def _load_tokenizer(self):
            _, tokenizer = load_rwkv_native(self.config.model_path)
            return tokenizer


    # 使用自定义管道
    pipeline = CustomPipeline(config)
    # ... 后续分析流程相同