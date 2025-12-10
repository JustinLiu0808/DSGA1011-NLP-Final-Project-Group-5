"""
RWKV 模型层级分析完整管道
整合激活分布分析与 Token 归因
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class AnalysisConfig:
    """分析配置"""
    model_path: str
    num_layers: int
    sensitive_layers: List[int] = None  # 如果为 None,自动从第一阶段确定
    top_k_ratio: float = 0.2
    ig_steps: int = 50
    deletion_steps: int = 10
    num_stability_trials: int = 3


class RWKVAnalysisPipeline:
    """RWKV 模型分析的端到端管道"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model = self._load_rwkv_model()
        self.tokenizer = self._load_tokenizer()

        # 初始化两个分析器
        from layer_activation_analyzer import LayerActivationAnalyzer
        from token_attribution_analyzer import TokenAttributionAnalyzer

        self.layer_analyzer = LayerActivationAnalyzer()
        self.token_analyzer = TokenAttributionAnalyzer(self.model, self.tokenizer)

        self.results = {}

    def _load_rwkv_model(self):
        """
        加载 RWKV 模型
        需要使用支持中间层输出的版本
        """
        # 方案 1: 使用 HuggingFace 版本
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                output_hidden_states=True  # 关键参数
            )
            return model
        except:
            pass

        # 方案 2: 使用原生 RWKV 库并修改
        try:
            from rwkv.model import RWKV
            model = RWKV(model=self.config.model_path, strategy='cpu fp32')
            # 需要 monkey-patch forward 方法以返回中间状态
            return self._wrap_rwkv_model(model)
        except:
            raise RuntimeError("无法加载 RWKV 模型,请检查路径和依赖")

    def _wrap_rwkv_model(self, rwkv_model):
        """
        包装原生 RWKV 模型以输出中间层状态
        这是关键步骤,需要根据实际 RWKV 实现调整
        """

        class RWKVWrapper(torch.nn.Module):
            def __init__(self, rwkv):
                super().__init__()
                self.rwkv = rwkv
                self.hidden_states = []

            def forward(self, input_ids, return_hidden_states=True):
                # 这里需要修改 RWKV 的前向传播逻辑
                # 伪代码示例:
                self.hidden_states = []
                state = None

                for token_id in input_ids[0]:
                    logits, state = self.rwkv.forward(int(token_id), state)
                    if return_hidden_states:
                        # 保存当前层的状态
                        self.hidden_states.append(state.copy())

                return type('Outputs', (), {
                    'logits': torch.tensor(logits).unsqueeze(0),
                    'hidden_states': self.hidden_states
                })()

            def get_input_embeddings(self):
                # 返回嵌入层
                return self.rwkv.w['emb.weight']

        return RWKVWrapper(rwkv_model)

    def _load_tokenizer(self):
        """加载分词器"""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.config.model_path)
        except:
            # 使用 RWKV 原生分词器
            from rwkv.utils import TOKENIZER
            return TOKENIZER("path/to/vocab.txt")

    def stage1_layer_sensitivity(self,
                                 short_texts: List[str],
                                 long_texts: List[str]) -> Dict:
        """
        阶段 1: 层级敏感度分析

        Args:
            short_texts: 短版本样本列表
            long_texts: 对应的长版本样本列表
        """
        print("=" * 60)
        print("阶段 1: 层级激活分布分析")
        print("=" * 60)

        # 对每对样本提取激活
        layer_activations_short = {i: [] for i in range(self.config.num_layers)}
        layer_activations_long = {i: [] for i in range(self.config.num_layers)}

        for idx, (short_txt, long_txt) in enumerate(zip(short_texts, long_texts)):
            print(f"处理样本对 {idx + 1}/{len(short_texts)}...")

            # 短文本
            short_ids = self.tokenizer(short_txt, return_tensors='pt')['input_ids']
            with torch.no_grad():
                short_outputs = self.model(short_ids, return_hidden_states=True)

            # 长文本
            long_ids = self.tokenizer(long_txt, return_tensors='pt')['input_ids']
            with torch.no_grad():
                long_outputs = self.model(long_ids, return_hidden_states=True)

            # 保存每层激活
            for layer_idx in range(self.config.num_layers):
                if hasattr(short_outputs, 'hidden_states'):
                    layer_activations_short[layer_idx].append(
                        short_outputs.hidden_states[layer_idx].cpu().numpy()
                    )
                    layer_activations_long[layer_idx].append(
                        long_outputs.hidden_states[layer_idx].cpu().numpy()
                    )

        # 计算分布指标
        layer_metrics = {}
        for layer_idx in range(self.config.num_layers):
            # 合并所有样本的激活
            short_acts = np.concatenate(layer_activations_short[layer_idx], axis=1)
            long_acts = np.concatenate(layer_activations_long[layer_idx], axis=1)

            metrics = self.layer_analyzer.compute_distribution_metrics(
                short_acts[0],
                long_acts[0]
            )
            layer_metrics[layer_idx] = metrics

            print(f"Layer {layer_idx}: Wasserstein={metrics['wasserstein']:.4f}, "
                  f"JSD={metrics['js_divergence']:.4f}")

        # 确定敏感层
        wass_scores = [layer_metrics[i]['wasserstein'] for i in range(self.config.num_layers)]
        max_wass = max(wass_scores)
        sensitivity_ranking = sorted(
            [(i, wass_scores[i] / max_wass) for i in range(self.config.num_layers)],
            key=lambda x: x[1],
            reverse=True
        )

        top_3_layers = [idx for idx, _ in sensitivity_ranking[:3]]

        summary = (f"层 {top_3_layers} 在长上下文下分布显著偏移,提示长程线索主要在这些层聚合。")

        print("\n" + summary)

        self.results['stage1'] = {
            'layer_metrics': layer_metrics,
            'sensitivity_ranking': sensitivity_ranking,
            'sensitive_layers': top_3_layers,
            'summary': summary
        }

        # 更新配置
        if self.config.sensitive_layers is None:
            self.config.sensitive_layers = top_3_layers

        return self.results['stage1']

    def stage2_token_attribution(self,
                                 long_text: str,
                                 target_idx: int,
                                 layer_idx: int = None) -> Dict:
        """
        阶段 2: Token 级归因分析

        Args:
            long_text: 单条长版本样本
            target_idx: 目标索引 (分类 logit 或生成位置)
            layer_idx: 指定层,如果为 None 则使用阶段1确定的敏感层
        """
        print("\n" + "=" * 60)
        print("阶段 2: Token 级归因与忠实度分析")
        print("=" * 60)

        if layer_idx is None and self.config.sensitive_layers:
            layer_idx = self.config.sensitive_layers[0]

        print(f"分析层: {layer_idx}")
        print(f"文本: {long_text[:100]}...")

        # 生成完整忠实度报告
        report = self.token_analyzer.full_fidelity_report(
            long_text,
            target_idx,
            layer_idx
        )

        print("\n" + report['summary'])

        # 可视化
        self.token_analyzer.visualize_heatmap(
            long_text,
            report['attributions'],
            title=f"Layer {layer_idx} Token Attribution",
            save_path=f"attribution_layer{layer_idx}.png"
        )

        self.results['stage2'] = report

        return report

    def run_full_analysis(self,
                          short_texts: List[str],
                          long_texts: List[str],
                          target_samples: List[Tuple[str, int]],
                          output_dir: str = "analysis_results") -> Dict:
        """
        运行完整分析流程

        Args:
            short_texts: 短文本列表 (用于阶段1)
            long_texts: 长文本列表 (用于阶段1)
            target_samples: [(长文本, 目标索引)] 列表 (用于阶段2)
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 阶段 1
        stage1_results = self.stage1_layer_sensitivity(short_texts, long_texts)

        # 绘制层级曲线
        self.layer_analyzer.plot_layer_curves(
            stage1_results['layer_metrics'],
            save_path=os.path.join(output_dir, "layer_curves.png")
        )

        # 阶段 2: 对每个敏感层分析样本
        stage2_results = []
        for layer_idx in self.config.sensitive_layers[:3]:  # 分析前 3 个敏感层
            print(f"\n{'=' * 60}\n分析敏感层 {layer_idx}\n{'=' * 60}")

            for text, target_idx in target_samples:
                result = self.stage2_token_attribution(text, target_idx, layer_idx)
                result['layer'] = layer_idx
                result['text'] = text[:100]
                stage2_results.append(result)

        self.results['stage2_all'] = stage2_results

        # 保存结果
        self._save_results(output_dir)

        # 生成综合报告
        self._generate_report(output_dir)

        return self.results

    def _save_results(self, output_dir: str):
        """保存结果到 JSON"""
        # 转换为可序列化格式
        serializable_results = {}

        if 'stage1' in self.results:
            serializable_results['stage1'] = {
                'layer_metrics': {
                    str(k): {kk: float(vv) for kk, vv in v.items()}
                    for k, v in self.results['stage1']['layer_metrics'].items()
                },
                'sensitivity_ranking': [
                    (int(idx), float(score))
                    for idx, score in self.results['stage1']['sensitivity_ranking']
                ],
                'summary': self.results['stage1']['summary']
            }

        if 'stage2_all' in self.results:
            serializable_results['stage2_all'] = [
                {
                    'layer': r['layer'],
                    'text': r['text'],
                    'deletion_auc': float(r['deletion_auc']),
                    'insertion_auc': float(r['insertion_auc']),
                    'comprehensiveness': float(r['comprehensiveness']),
                    'sufficiency': float(r['sufficiency']),
                    'stability': float(r['stability']),
                    'top_tokens': r['top_tokens'],
                    'summary': r['summary']
                }
                for r in self.results['stage2_all']
            ]

        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def _generate_report(self, output_dir: str):
        """生成 Markdown 格式报告"""
        report_lines = [
            "# RWKV 模型层级分析报告\n",
            f"## 配置\n",
            f"- 模型: {self.config.model_path}",
            f"- 总层数: {self.config.num_layers}",
            f"- 敏感层: {self.config.sensitive_layers}\n",
            "---\n",
            "## 阶段 1: 层级激活分布分析\n"
        ]

        if 'stage1' in self.results:
            report_lines.append(f"### 结论\n{self.results['stage1']['summary']}\n")
            report_lines.append("\n### 敏感度排名 (Top 5)\n")
            for idx, (layer, score) in enumerate(self.results['stage1']['sensitivity_ranking'][:5]):
                report_lines.append(f"{idx + 1}. Layer {layer}: {score:.4f}")
            report_lines.append("\n![层级曲线](layer_curves.png)\n")

        report_lines.append("\n---\n## 阶段 2: Token 级归因分析\n")

        if 'stage2_all' in self.results:
            for result in self.results['stage2_all']:
                report_lines.append(f"\n### Layer {result['layer']}\n")
                report_lines.append(f"**文本**: {result['text']}...\n")
                report_lines.append(f"**高热 Token**: {', '.join(result['top_tokens'])}\n")
                report_lines.append(f"\n**忠实度指标**:")
                report_lines.append(f"- Deletion AUC: {result['deletion_auc']:.4f}")
                report_lines.append(f"- Insertion AUC: {result['insertion_auc']:.4f}")
                report_lines.append(f"- Comprehensiveness: {result['comprehensiveness']:.4f}")
                report_lines.append(f"- Sufficiency: {result['sufficiency']:.4f}")
                report_lines.append(f"- Stability (ρ): {result['stability']:.4f}\n")
                report_lines.append(f"![归因热图](attribution_layer{result['layer']}.png)\n")

        with open(os.path.join(output_dir, 'REPORT.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n报告已保存至 {output_dir}/REPORT.md")


# 使用示例
if __name__ == "__main__":
    # 配置
    config = AnalysisConfig(
        model_path="path/to/rwkv-model",
        num_layers=12,
        top_k_ratio=0.2,
        ig_steps=50
    )

    # 创建管道
    pipeline = RWKVAnalysisPipeline(config)

    # 准备数据
    short_texts = [
        "这是短文本示例1",
        "短样本2",
        # ...
    ]

    long_texts = [
        "这是长文本示例1,包含大量额外上下文信息..." * 10,
        "长样本2,有更多内容..." * 10,
        # ...
    ]

    target_samples = [
        ("要分析的长文本1...", 0),  # (文本, 目标类别/位置)
        ("要分析的长文本2...", 1),
    ]

    # 运行完整分析
    results = pipeline.run_full_analysis(
        short_texts,
        long_texts,
        target_samples,
        output_dir="rwkv_analysis_output"
    )