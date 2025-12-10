import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon, wasserstein_distance
from typing import Dict, List, Tuple
import requests
import json


class LayerActivationAnalyzer:
    """分析模型层级激活分布,对比短文本与长文本"""

    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url
        self.layer_stats = {}

    def extract_activations(self, text: str, return_hidden_states: bool = True) -> Dict:
        """
        通过 RWKV-Runner API 提取激活
        注意: RWKV-Runner 可能需要修改以返回中间层状态
        """
        payload = {
            "text": text,
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False
        }

        # 如果 API 支持,添加参数以返回隐藏层
        if return_hidden_states:
            payload["output_hidden_states"] = True

        response = requests.post(f"{self.api_url}/v1/completions", json=payload)
        return response.json()

    def compute_distribution_metrics(self,
                                     short_acts: np.ndarray,
                                     long_acts: np.ndarray) -> Dict[str, float]:
        """
        计算短文本与长文本激活的分布差异

        Args:
            short_acts: shape [seq_len_short, hidden_dim]
            long_acts: shape [seq_len_long, hidden_dim]
        """
        metrics = {}

        # 展平为一维分布(或沿维度聚合)
        short_flat = short_acts.flatten()
        long_flat = long_acts.flatten()

        # 1. KS 检验 (Kolmogorov-Smirnov)
        ks_stat, ks_pval = stats.ks_2samp(short_flat, long_flat)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pval

        # 2. Wasserstein 距离 (Earth Mover's Distance)
        # 需要先做分箱或采样
        short_sample = np.random.choice(short_flat, size=min(10000, len(short_flat)), replace=False)
        long_sample = np.random.choice(long_flat, size=min(10000, len(long_flat)), replace=False)
        metrics['wasserstein'] = wasserstein_distance(short_sample, long_sample)

        # 3. Jensen-Shannon 散度
        # 构建直方图
        bins = np.linspace(
            min(short_flat.min(), long_flat.min()),
            max(short_flat.max(), long_flat.max()),
            100
        )
        short_hist, _ = np.histogram(short_flat, bins=bins, density=True)
        long_hist, _ = np.histogram(long_flat, bins=bins, density=True)
        short_hist = short_hist + 1e-10  # 避免零
        long_hist = long_hist + 1e-10
        short_hist /= short_hist.sum()
        long_hist /= long_hist.sum()
        metrics['js_divergence'] = jensenshannon(short_hist, long_hist)

        # 4. 稀疏度 (激活率)
        threshold = 0.1  # 可调
        metrics['sparsity_short'] = (np.abs(short_flat) > threshold).mean()
        metrics['sparsity_long'] = (np.abs(long_flat) > threshold).mean()
        metrics['sparsity_diff'] = metrics['sparsity_long'] - metrics['sparsity_short']

        # 5. 熵与方差
        metrics['entropy_short'] = stats.entropy(short_hist)
        metrics['entropy_long'] = stats.entropy(long_hist)
        metrics['variance_short'] = np.var(short_flat)
        metrics['variance_long'] = np.var(long_flat)

        return metrics

    def compute_attention_distance(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """
        计算注意力的平均距离与远距占比

        Args:
            attention_weights: shape [num_heads, seq_len, seq_len]
        """
        num_heads, seq_len, _ = attention_weights.shape

        # 创建距离矩阵
        positions = np.arange(seq_len)
        distance_matrix = np.abs(positions[:, None] - positions[None, :])

        # 平均注意力距离
        avg_distances = []
        long_range_ratios = []

        for head in range(num_heads):
            attn = attention_weights[head]
            avg_dist = (attn * distance_matrix).sum() / attn.sum()
            avg_distances.append(avg_dist)

            # 远距注意力 (距离 > seq_len/4)
            long_range_mask = distance_matrix > (seq_len / 4)
            long_range_ratio = attn[long_range_mask].sum() / attn.sum()
            long_range_ratios.append(long_range_ratio)

        return {
            'avg_attention_distance': np.mean(avg_distances),
            'long_range_attention_ratio': np.mean(long_range_ratios),
            'per_head_avg_distance': avg_distances
        }

    def analyze_layer_sensitivity(self,
                                  short_texts: List[str],
                                  long_texts: List[str],
                                  num_layers: int) -> Dict:
        """
        批量对比短文本与长文本,生成每层敏感度排名

        Returns:
            {
                'layer_metrics': {layer_idx: metrics_dict},
                'sensitivity_ranking': [(layer_idx, score), ...],
                'summary': str
            }
        """
        layer_metrics = {i: [] for i in range(num_layers)}

        # 处理每对样本
        for short_txt, long_txt in zip(short_texts, long_texts):
            # 这里需要根据实际 API 返回结构调整
            # 假设可以获取 hidden_states: List[np.ndarray]
            short_hidden = self._get_hidden_states(short_txt)
            long_hidden = self._get_hidden_states(long_txt)

            for layer_idx in range(num_layers):
                if layer_idx < len(short_hidden) and layer_idx < len(long_hidden):
                    metrics = self.compute_distribution_metrics(
                        short_hidden[layer_idx],
                        long_hidden[layer_idx]
                    )
                    layer_metrics[layer_idx].append(metrics)

        # 聚合指标并排名
        aggregated = {}
        for layer_idx, metrics_list in layer_metrics.items():
            agg = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                agg[key] = np.mean(values)
            aggregated[layer_idx] = agg

        # 计算敏感度分数 (归一化 Wasserstein 距离)
        wass_scores = [aggregated[i]['wasserstein'] for i in range(num_layers)]
        max_wass = max(wass_scores) if max(wass_scores) > 0 else 1
        sensitivity_ranking = sorted(
            [(i, wass_scores[i] / max_wass) for i in range(num_layers)],
            key=lambda x: x[1],
            reverse=True
        )

        # 生成结论
        top_layers = [idx for idx, _ in sensitivity_ranking[:3]]
        summary = (f"层 {top_layers} 在长上下文下分布显著偏移 "
                   f"(归一化 Wasserstein: {[f'{aggregated[i][\"wasserstein\"]/max_wass:.3f}' for i in top_layers]}), "
                   f"提示长程线索主要在这些层聚合。")

        return {
            'layer_metrics': aggregated,
            'sensitivity_ranking': sensitivity_ranking,
            'summary': summary
        }

    def _get_hidden_states(self, text: str) -> List[np.ndarray]:
        """
        辅助函数: 从 API 提取隐藏状态
        需要根据 RWKV-Runner 实际实现调整
        """
        # 占位实现 - 实际需要修改 RWKV-Runner 以输出中间层
        # 可能需要直接加载模型进行前向传播
        raise NotImplementedError(
            "需要修改 RWKV-Runner 或直接加载 RWKV 模型以提取 hidden_states"
        )

    def plot_layer_curves(self, layer_metrics: Dict, save_path: str = None):
        """绘制各层指标曲线"""
        import matplotlib.pyplot as plt

        layers = sorted(layer_metrics.keys())
        metrics_to_plot = ['wasserstein', 'js_divergence', 'sparsity_diff']

        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 8))

        for idx, metric in enumerate(metrics_to_plot):
            values = [layer_metrics[l][metric] for l in layers]
            axes[idx].plot(layers, values, marker='o')
            axes[idx].set_ylabel(metric)
            axes[idx].set_xlabel('Layer')
            axes[idx].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    analyzer = LayerActivationAnalyzer()

    # 准备数据
    short_texts = ["这是短文本示例"] * 10
    long_texts = ["这是长文本示例" + "额外内容" * 50] * 10

    # 分析
    # results = analyzer.analyze_layer_sensitivity(short_texts, long_texts, num_layers=12)
    # print(results['summary'])
    # analyzer.plot_layer_curves(results['layer_metrics'])