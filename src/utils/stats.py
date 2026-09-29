"""
Statistical Analysis Module for KG Debugging
=============================================
Comprehensive statistical evaluation and comparison of detection methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, mannwhitneyu
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


class StatisticalEvaluator:
    """
    Complete statistical analysis for KG anomaly detection
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha

    def compare_detection_methods(self,
                                  method_results: Dict[str, List[Dict]],
                                  ground_truth: Optional[List[bool]] = None) -> Dict:
        """
        Compare multiple detection methods statistically

        Args:
            method_results: Dict mapping method name to list of detection results
            ground_truth: Optional list of true anomaly labels

        Returns:
            Comprehensive comparison results
        """
        results = {
            'basic_stats': self.compute_basic_statistics(method_results),
            'agreement': self.compute_method_agreement(method_results),
            'correlation': self.compute_method_correlation(method_results),
            'statistical_tests': self.perform_statistical_tests(method_results),
        }

        if ground_truth is not None:
            results['performance'] = self.evaluate_performance(method_results, ground_truth)

        results['summary'] = self.generate_summary(results)

        return results

    def compute_basic_statistics(self, method_results: Dict[str, List[Dict]]) -> Dict:
        """Compute basic statistics for each method"""
        stats_dict = {}

        for method, results in method_results.items():
            scores = [r.get('score', 0.0) for r in results]

            if scores:
                stats_dict[method] = {
                    'n_detections': len(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'median_score': np.median(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'q25': np.percentile(scores, 25),
                    'q75': np.percentile(scores, 75),
                    'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                    'skewness': stats.skew(scores),
                    'kurtosis': stats.kurtosis(scores)
                }
            else:
                stats_dict[method] = {
                    'n_detections': 0,
                    'mean_score': 0.0,
                    'std_score': 0.0
                }

        return stats_dict

    def compute_method_agreement(self, method_results: Dict[str, List[Dict]]) -> Dict:
        """
        Compute agreement between detection methods
        """
        methods = list(method_results.keys())
        n_methods = len(methods)

        # Create detection matrices for each method
        all_triples = set()
        for results in method_results.values():
            for r in results:
                triple_key = (r['triple'].subject, r['triple'].relation, r['triple'].object)
                all_triples.add(triple_key)

        all_triples = list(all_triples)
        n_triples = len(all_triples)
      
        # Binary detection matrix
        detection_matrix = np.zeros((n_methods, n_triples))

        for i, method in enumerate(methods):
            detected = set()
            for r in method_results[method]:
                triple_key = (r['triple'].subject, r['triple'].relation, r['triple'].object)
                detected.add(triple_key)

            for j, triple in enumerate(all_triples):
                if triple in detected:
                    detection_matrix[i, j] = 1

        # Compute agreement metrics
        agreement_results = {
            'methods': methods,
            'n_triples': n_triples,
            'detection_matrix_shape': detection_matrix.shape,
            'pairwise_agreement': {},
            'multi_method_agreement': {}
        }

        # Pairwise agreement (Cohen's kappa)
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                try:
                    kappa = cohen_kappa_score(detection_matrix[i], detection_matrix[j])
                    agreement_results['pairwise_agreement'][f"{methods[i]}_vs_{methods[j]}"] = {
                        'kappa': kappa,
                        'interpretation': self._interpret_kappa(kappa)
                    }
                except:
                    agreement_results['pairwise_agreement'][f"{methods[i]}_vs_{methods[j]}"] = {
                        'kappa': 0.0,
                        'interpretation': 'Could not compute'
                    }

        # Multi-method agreement (Fleiss' kappa approximation)
        try:
            # Count how many methods agree on each triple
            agreement_counts = np.sum(detection_matrix, axis=0)

            # Percentage of triples detected by k methods
            for k in range(n_methods + 1):
                count = np.sum(agreement_counts == k)
                percentage = count / n_triples * 100
                agreement_results['multi_method_agreement'][f'detected_by_{k}_methods'] = {
                    'count': int(count),
                    'percentage': percentage
                }
        except:
            pass

        # Jaccard similarity between methods
        agreement_results['jaccard_similarity'] = {}
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                intersection = np.sum((detection_matrix[i] == 1) & (detection_matrix[j] == 1))
                union = np.sum((detection_matrix[i] == 1) | (detection_matrix[j] == 1))
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 0.0
                agreement_results['jaccard_similarity'][f"{methods[i]}_vs_{methods[j]}"] = jaccard

        return agreement_results

    def compute_method_correlation(self, method_results: Dict[str, List[Dict]]) -> Dict:
        """Compute correlation between method scores"""
        methods = list(method_results.keys())

        # Align scores for common triples
        common_triples = set()
        for results in method_results.values():
            for r in results:
                triple_key = (r['triple'].subject, r['triple'].relation, r['triple'].object)
                common_triples.add(triple_key)

        common_triples = list(common_triples)

        # Create score matrix
        score_matrix = {}
        for method in methods:
            scores = {}
            for r in method_results[method]:
                triple_key = (r['triple'].subject, r['triple'].relation, r['triple'].object)
                scores[triple_key] = r.get('score', 0.0)

            score_matrix[method] = [scores.get(t, 0.0) for t in common_triples]

        # Compute correlations
        correlation_results = {
            'pearson': {},
            'spearman': {},
            'kendall': {}
        }

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    scores1 = score_matrix[method1]
                    scores2 = score_matrix[method2]

                    if len(scores1) > 2 and len(scores2) > 2:
                        # Pearson correlation
                        try:
                            pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
                            correlation_results['pearson'][f"{method1}_vs_{method2}"] = {
                                'r': pearson_r,
                                'p_value': pearson_p,
                                'significant': pearson_p < self.alpha
                            }
                        except:
                            pass

                        # Spearman correlation
                        try:
                            spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
                            correlation_results['spearman'][f"{method1}_vs_{method2}"] = {
                                'rho': spearman_r,
                                'p_value': spearman_p,
                                'significant': spearman_p < self.alpha
                            }
                        except:
                            pass

                        # Kendall correlation
                        try:
                            kendall_tau, kendall_p = stats.kendalltau(scores1, scores2)
                            correlation_results['kendall'][f"{method1}_vs_{method2}"] = {
                                'tau': kendall_tau,
                                'p_value': kendall_p,
                                'significant': kendall_p < self.alpha
                            }
                        except:
                            pass

        return correlation_results

    def perform_statistical_tests(self, method_results: Dict[str, List[Dict]]) -> Dict:
        """Perform statistical significance tests"""
        methods = list(method_results.keys())
        n_methods = len(methods)

        test_results = {
            'n_methods': n_methods,
            'methods': methods
        }

        if n_methods < 2:
            test_results['message'] = "Need at least 2 methods for comparison"
            return test_results

        # Prepare scores for each method
        method_scores = {}
        for method, results in method_results.items():
            method_scores[method] = [r.get('score', 0.0) for r in results]

        # Ensure all methods have same number of scores (pad with zeros)
        max_len = max(len(scores) for scores in method_scores.values())
        for method in methods:
            while len(method_scores[method]) < max_len:
                method_scores[method].append(0.0)

        if n_methods == 2:
            # Two methods: Use Wilcoxon signed-rank test or Mann-Whitney U
            method1, method2 = methods
            scores1 = method_scores[method1]
            scores2 = method_scores[method2]

            # Wilcoxon signed-rank (paired)
            try:
                stat, p_value = wilcoxon(scores1, scores2)
                test_results['wilcoxon'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'interpretation': 'Methods differ significantly' if p_value < self.alpha else 'No significant difference'
                }
            except:
                test_results['wilcoxon'] = {'error': 'Could not compute'}

            # Mann-Whitney U (unpaired)
            try:
                stat, p_value = mannwhitneyu(scores1, scores2)
                test_results['mann_whitney'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            except:
                test_results['mann_whitney'] = {'error': 'Could not compute'}

            # Effect size (Cohen's d)
            try:
                mean_diff = np.mean(scores1) - np.mean(scores2)
                pooled_std = np.sqrt((np.std(scores1) ** 2 + np.std(scores2) ** 2) / 2)
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                else:
                    cohens_d = 0.0

                test_results['effect_size'] = {
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
            except:
                pass

        elif n_methods > 2:
            # Multiple methods: Use Friedman test
            try:
                score_matrix = np.array([method_scores[m] for m in methods]).T
                stat, p_value = friedmanchisquare(*score_matrix.T)

                test_results['friedman'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'interpretation': 'Methods differ significantly' if p_value < self.alpha else 'No significant difference'
                }

                # If significant, perform post-hoc tests
                if p_value < self.alpha:
                    test_results['post_hoc'] = {}
                    for i in range(n_methods):
                        for j in range(i + 1, n_methods):
                            try:
                                stat, p = wilcoxon(method_scores[methods[i]],
                                                   method_scores[methods[j]])
                                # Bonferroni correction
                                adjusted_p = p * (n_methods * (n_methods - 1) / 2)
                                test_results['post_hoc'][f"{methods[i]}_vs_{methods[j]}"] = {
                                    'p_value': p,
                                    'adjusted_p_value': adjusted_p,
                                    'significant': adjusted_p < self.alpha
                                }
                            except:
                                pass
            except:
                test_results['friedman'] = {'error': 'Could not compute'}

        return test_results

    def evaluate_performance(self, method_results: Dict[str, List[Dict]],
                             ground_truth: List[bool]) -> Dict:
        """Evaluate performance against ground truth"""
        performance = {}

        for method, results in method_results.items():
            predictions = [r.get('score', 0.0) > 0.5 for r in results]

            if len(predictions) == len(ground_truth):
                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

                # Metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                # ROC-AUC if we have scores
                scores = [r.get('score', 0.0) for r in results]
                try:
                    auc = roc_auc_score(ground_truth, scores)
                except:
                    auc = 0.0

                performance[method] = {
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'auc_roc': auc
                }

        return performance

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's kappa value"""
        if kappa < 0:
            return "Poor agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    def generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary"""
        lines = ["=" * 60, "STATISTICAL ANALYSIS SUMMARY", "=" * 60]

        # Basic statistics
        if 'basic_stats' in results:
            lines.append("\n1. DETECTION STATISTICS:")
            for method, stats in results['basic_stats'].items():
                lines.append(f"  {method}:")
                lines.append(f"    - Detections: {stats['n_detections']}")
                lines.append(f"    - Mean score: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")

        # Agreement
        if 'agreement' in results and 'pairwise_agreement' in results['agreement']:
            lines.append("\n2. METHOD AGREEMENT:")
            for pair, agreement in results['agreement']['pairwise_agreement'].items():
                lines.append(f"  {pair}: κ = {agreement['kappa']:.3f} ({agreement['interpretation']})")

        # Statistical tests
        if 'statistical_tests' in results:
            lines.append("\n3. STATISTICAL TESTS:")
            tests = results['statistical_tests']

            if 'friedman' in tests:
                lines.append(f"  Friedman test: p = {tests['friedman']['p_value']:.4f}")
                lines.append(f"  → {tests['friedman']['interpretation']}")
            elif 'wilcoxon' in tests:
                lines.append(f"  Wilcoxon test: p = {tests['wilcoxon']['p_value']:.4f}")
                lines.append(f"  → {tests['wilcoxon']['interpretation']}")

            if 'effect_size' in tests:
                lines.append(f"  Effect size (Cohen's d): {tests['effect_size']['cohens_d']:.3f}")
                lines.append(f"  → {tests['effect_size']['interpretation']}")

        # Performance (if available)
        if 'performance' in results:
            lines.append("\n4. PERFORMANCE METRICS:")
            for method, perf in results['performance'].items():
                lines.append(f"  {method}:")
                lines.append(f"    - Precision: {perf['precision']:.3f}")
                lines.append(f"    - Recall: {perf['recall']:.3f}")
                lines.append(f"    - F1: {perf['f1_score']:.3f}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_latex_table(self, results: Dict, caption: str = "Statistical Comparison") -> str:
        """Generate LaTeX table for paper"""
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")

        # Determine which table to generate based on results
        if 'basic_stats' in results:
            lines.append("\\begin{tabular}{l|cccc}")
            lines.append("\\hline")
            lines.append("Method & Detections & Mean Score & Std & Median \\\\")
            lines.append("\\hline")

            for method, stats in results['basic_stats'].items():
                lines.append(f"{method} & {stats['n_detections']} & "
                             f"{stats['mean_score']:.3f} & {stats['std_score']:.3f} & "
                             f"{stats['median_score']:.3f} \\\\")

            lines.append("\\hline")
            lines.append("\\end{tabular}")

        lines.append("\\label{tab:statistical_comparison}")
        lines.append("\\end{table}")

        return "\n".join(lines)


# Utility functions
def compare_anomaly_detectors(ode_results: List, pykeen_results: List) -> Dict:
    """
    Compare ODE and PyKEEN anomaly detectors

    Args:
        ode_results: List of ODE detection results
        pykeen_results: List of PyKEEN detection results

    Returns:
        Comparison results
    """
    evaluator = StatisticalEvaluator()

    # Convert to standard format
    method_results = {
        'ODE': [{'triple': r.triple, 'score': r.anomaly_score} for r in ode_results],
        'PyKEEN': [{'triple': r.triple, 'score': r.anomaly_score} for r in pykeen_results]
    }

    return evaluator.compare_detection_methods(method_results)


if __name__ == "__main__":
    # Test the statistics module
    print("Testing Statistical Evaluator...")

    # Create mock results
    from pykeen_detector import Triple

    # Mock ODE results
    ode_results = [
        {'triple': Triple('A', 'r', 'B'), 'score': 0.8},
        {'triple': Triple('C', 'r', 'D'), 'score': 0.6},
        {'triple': Triple('E', 'r', 'F'), 'score': 0.9},
    ]

    # Mock PyKEEN results
    pykeen_results = [
        {'triple': Triple('A', 'r', 'B'), 'score': 0.7},
        {'triple': Triple('C', 'r', 'D'), 'score': 0.8},
        {'triple': Triple('G', 'r', 'H'), 'score': 0.5},
    ]

    evaluator = StatisticalEvaluator()
    results = evaluator.compare_detection_methods({
        'ODE': ode_results,
        'PyKEEN': pykeen_results
    })

    print(results['summary'])
