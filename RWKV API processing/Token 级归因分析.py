import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns


class TokenAttributionAnalyzer:
    """Token 级归因分析与忠实度评估"""

    def __init__(self, model, tokenizer):
        """
        Args:
            model: RWKV 模型实例
            tokenizer: 对应的分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def integrated_gradients(self,
                             input_ids: torch.Tensor,
                             target_idx: int,
                             layer_idx: int = None,
                             n_steps: int = 50) -> np.ndarray:
        """
        使用 Integrated Gradients 计算归因

        Args:
            input_ids: shape [1, seq_len]
            target_idx: 目标 token 位置或分类 logit 索引
            layer_idx: 如果指定,计算特定层的归因

        Returns:
            attributions: shape [seq_len, hidden_dim] 或 [seq_len]
        """
        # 获取嵌入层
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad = True

        # 定义前向函数
        def forward_func(embeds):
            outputs = self.model(inputs_embeds=embeds)
            if isinstance(target_idx, int):
                # 分类任务: 返回目标类别 logit
                return outputs.logits[:, target_idx]
            else:
                # 生成任务: 返回目标位置概率
                return outputs.logits[:, target_idx[0], target_idx[1]]

        # 计算 IG
        ig = IntegratedGradients(forward_func)

        # 基线: 零嵌入
        baseline = torch.zeros_like(embeddings)

        attributions = ig.attribute(
            embeddings,
            baseline,
            n_steps=n_steps,
            internal_batch_size=1
        )

        # 按 L2 范数聚合到 token 级
        token_attrs = attributions.squeeze(0).norm(dim=1).cpu().detach().numpy()

        return token_attrs

    def gradient_based_attribution(self,
                                   input_ids: torch.Tensor,
                                   target_idx: int) -> np.ndarray:
        """简单梯度法 (作为对比基线)"""
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad = True

        outputs = self.model(inputs_embeds=embeddings)

        if isinstance(target_idx, int):
            target_score = outputs.logits[:, target_idx]
        else:
            target_score = outputs.logits[:, target_idx[0], target_idx[1]]

        target_score.backward()

        # 梯度 × 输入
        attrs = (embeddings.grad * embeddings).sum(dim=-1).squeeze(0)
        return attrs.abs().cpu().numpy()

    def deletion_curve(self,
                       input_ids: torch.Tensor,
                       attributions: np.ndarray,
                       target_idx: int,
                       steps: int = 10) -> Tuple[List[float], float]:
        """
        按归因分数递减删除 token,记录目标分数变化

        Returns:
            scores: 每步删除后的目标分数
            auc: 曲线下面积 (归一化)
        """
        seq_len = input_ids.size(1)
        sorted_indices = np.argsort(attributions)[::-1]  # 从高到低

        scores = []
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id

        for step in range(steps + 1):
            # 删除前 step/steps 比例的高归因 token
            num_to_mask = int((step / steps) * seq_len)
            masked_ids = input_ids.clone()

            if num_to_mask > 0:
                mask_positions = sorted_indices[:num_to_mask]
                masked_ids[0, mask_positions] = mask_token_id

            # 计算分数
            with torch.no_grad():
                outputs = self.model(masked_ids)
                if isinstance(target_idx, int):
                    score = outputs.logits[0, target_idx].item()
                else:
                    score = outputs.logits[0, target_idx[0], target_idx[1]].item()

            scores.append(score)

        # 计算 AUC (梯形法则)
        auc = np.trapz(scores, dx=1.0 / steps) / scores[0] if scores[0] != 0 else 0

        return scores, auc

    def insertion_curve(self,
                        input_ids: torch.Tensor,
                        attributions: np.ndarray,
                        target_idx: int,
                        steps: int = 10) -> Tuple[List[float], float]:
        """
        从全遮蔽开始,按归因分数递减插入 token
        """
        seq_len = input_ids.size(1)
        sorted_indices = np.argsort(attributions)[::-1]

        scores = []
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id

        for step in range(steps + 1):
            num_to_reveal = int((step / steps) * seq_len)
            masked_ids = torch.full_like(input_ids, mask_token_id)

            if num_to_reveal > 0:
                reveal_positions = sorted_indices[:num_to_reveal]
                masked_ids[0, reveal_positions] = input_ids[0, reveal_positions]

            with torch.no_grad():
                outputs = self.model(masked_ids)
                if isinstance(target_idx, int):
                    score = outputs.logits[0, target_idx].item()
                else:
                    score = outputs.logits[0, target_idx[0], target_idx[1]].item()

            scores.append(score)

        original_score = scores[-1] if len(scores) > 0 else 1.0
        auc = np.trapz(scores, dx=1.0 / steps) / original_score if original_score != 0 else 0

        return scores, auc

    def comprehensiveness_sufficiency(self,
                                      input_ids: torch.Tensor,
                                      attributions: np.ndarray,
                                      target_idx: int,
                                      top_k_ratio: float = 0.2) -> Dict[str, float]:
        """
        ERASER 风格的 Comprehensiveness & Sufficiency

        Comprehensiveness: 删除高热 token 导致的分数下降
        Sufficiency: 只保留高热 token 时保住的分数
        """
        seq_len = input_ids.size(1)
        top_k = max(1, int(seq_len * top_k_ratio))
        top_indices = np.argsort(attributions)[-top_k:]

        # 原始分数
        with torch.no_grad():
            outputs = self.model(input_ids)
            if isinstance(target_idx, int):
                original_score = outputs.logits[0, target_idx].item()
            else:
                original_score = outputs.logits[0, target_idx[0], target_idx[1]].item()

        # Comprehensiveness: 删除高热 token
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id
        masked_ids = input_ids.clone()
        masked_ids[0, top_indices] = mask_token_id

        with torch.no_grad():
            outputs = self.model(masked_ids)
            if isinstance(target_idx, int):
                comp_score = outputs.logits[0, target_idx].item()
            else:
                comp_score = outputs.logits[0, target_idx[0], target_idx[1]].item()

        comprehensiveness = original_score - comp_score

        # Sufficiency: 只保留高热 token
        suff_ids = torch.full_like(input_ids, mask_token_id)
        suff_ids[0, top_indices] = input_ids[0, top_indices]

        with torch.no_grad():
            outputs = self.model(suff_ids)
            if isinstance(target_idx, int):
                suff_score = outputs.logits[0, target_idx].item()
            else:
                suff_score = outputs.logits[0, target_idx[0], target_idx[1]].item()

        sufficiency = original_score - suff_score

        return {
            'comprehensiveness': comprehensiveness,
            'sufficiency': sufficiency,
            'original_score': original_score,
            'comp_score': comp_score,
            'suff_score': suff_score
        }

    def stability_test(self,
                       input_ids: torch.Tensor,
                       target_idx: int,
                       num_trials: int = 5,
                       paraphrase_fn=None) -> float:
        """
        稳定性测试: 多次运行或同义改写后的热图秩相关

        Args:
            paraphrase_fn: 同义改写函数 (可选)

        Returns:
            mean_spearman: 平均 Spearman 相关系数
        """
        attributions_list = []

        for trial in range(num_trials):
            if paraphrase_fn is not None and trial > 0:
                # 应用同义改写
                text = self.tokenizer.decode(input_ids[0])
                paraphrased = paraphrase_fn(text)
                input_ids_trial = self.tokenizer(paraphrased, return_tensors='pt')['input_ids']
            else:
                input_ids_trial = input_ids

            attrs = self.integrated_gradients(input_ids_trial, target_idx)
            attributions_list.append(attrs)

        # 计算两两 Spearman 相关
        from scipy.stats import spearmanr
        correlations = []
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                # 需要长度对齐
                min_len = min(len(attributions_list[i]), len(attributions_list[j]))
                corr, _ = spearmanr(
                    attributions_list[i][:min_len],
                    attributions_list[j][:min_len]
                )
                correlations.append(corr)

        return np.mean(correlations) if correlations else 1.0

    def visualize_heatmap(self,
                          text: str,
                          attributions: np.ndarray,
                          title: str = "Token Attribution Heatmap",
                          save_path: str = None):
        """可视化热力图"""
        tokens = self.tokenizer.tokenize(text)

        # 归一化归因到 [0, 1]
        attrs_norm = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(12, 2))

        # 使用 imshow 绘制
        im = ax.imshow([attrs_norm], cmap='Reds', aspect='auto')

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title(title)

        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def full_fidelity_report(self,
                             text: str,
                             target_idx: int,
                             layer_idx: int = None) -> Dict:
        """
        生成完整忠实度报告

        Returns:
            {
                'attributions': np.ndarray,
                'deletion_auc': float,
                'insertion_auc': float,
                'comprehensiveness': float,
                'sufficiency': float,
                'stability': float,
                'summary': str
            }
        """
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)

        # 计算归因
        attributions = self.integrated_gradients(input_ids, target_idx, layer_idx)

        # 忠实度指标
        _, del_auc = self.deletion_curve(input_ids, attributions, target_idx)
        _, ins_auc = self.insertion_curve(input_ids, attributions, target_idx)

        cs_metrics = self.comprehensiveness_sufficiency(input_ids, attributions, target_idx)

        stability = self.stability_test(input_ids, target_idx, num_trials=3)

        # 找出高热 token
        top_k = 5
        top_indices = np.argsort(attributions)[-top_k:][::-1]
        tokens = self.tokenizer.tokenize(text)
        top_tokens = [tokens[i] for i in top_indices if i < len(tokens)]

        summary = (
            f"在第 {layer_idx if layer_idx else '输出'} 层,高热 token 集中在: {top_tokens}; "
            f"按热度遮蔽导致 Δscore={cs_metrics['comprehensiveness']:.3f}, "
            f"说明该层确实由这些 token 驱动。"
            f"删除 AUC={del_auc:.3f}, 插入 AUC={ins_auc:.3f}, 稳定性 ρ={stability:.3f}。"
        )

        return {
            'attributions': attributions,
            'deletion_auc': del_auc,
            'insertion_auc': ins_auc,
            'comprehensiveness': cs_metrics['comprehensiveness'],
            'sufficiency': cs_metrics['sufficiency'],
            'stability': stability,
            'top_tokens': top_tokens,
            'summary': summary
        }


# 使用示例
if __name__ == "__main__":
    # 加载模型和分词器
    # model = load_rwkv_model(...)
    # tokenizer = load_tokenizer(...)

    # analyzer = TokenAttributionAnalyzer(model, tokenizer)

    # text = "这是一段需要分析的长文本..."
    # target_idx = 0  # 分类任务的目标类别

    # report = analyzer.full_fidelity_report(text, target_idx, layer_idx=8)
    # print(report['summary'])
    # analyzer.visualize_heatmap(text, report['attributions'])
    pass