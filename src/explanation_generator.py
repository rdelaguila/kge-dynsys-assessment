"""
Explanation Generator
=====================
Generates comprehensive analysis and explanations for experiments
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set


def load_detections(base_dir: Path, method: str) -> List[Dict]:
    """Load detections for a specific method"""
    det_file = base_dir / method / 'detections.json'
    if det_file.exists():
        with open(det_file, 'r') as f:
            return json.load(f)
    return []


def load_method_results(base_dir: Path, method: str) -> Dict:
    """Load results for a specific method"""
    results_file = base_dir / method / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def generate_dataset_explanations(base_dir: Path, output_dir: Path):
    """Generate explanations and analysis"""
    
    # Auto-detect available methods from directory structure
    methods = []
    excluded_dirs = {'ablation', 'visualizations', 'explanations'}
    for item in base_dir.iterdir():
        if item.is_dir() and item.name not in excluded_dirs:
            if (item / 'results.json').exists():
                methods.append(item.name)
    
    if not methods:
        print("No method results found in", base_dir)
        return
    
    print(f"Found methods: {methods}")
    results = {}
    detections = {}
    
    # Load all data
    for method in methods:
        res = load_method_results(base_dir, method)
        det = load_detections(base_dir, method)
        if res:
            results[method] = res
            detections[method] = det
    
    if not results:
        print("No results found to analyze")
        return
    
    # Generate global analysis
    _generate_global_analysis(results, detections, output_dir)
    
    # Generate overlap analysis
    _generate_overlap_analysis(detections, output_dir)
    
    # Generate detailed report
    _generate_detailed_report(results, detections, output_dir)
    
    # Generate expert summary
    _generate_expert_summary(results, detections, output_dir)
    
    # Generate ablation summary
    _generate_ablation_summary(base_dir, output_dir)

def _generate_ablation_summary(base_dir: Path, output_dir: Path):
    """Summarize ablation metrics"""
    metrics_dir = base_dir / 'metrics'
    if not metrics_dir.exists():
        return
        
    lines = []
    lines.append("ABLATION ANALYSIS REPORT")
    lines.append("========================")
    lines.append("")
    
    # Load all metrics
    ablation_data = []
    for f in metrics_dir.glob("metrics_*.json"):
        with open(f, 'r') as file:
            try:
                data = json.load(file)
                ablation_data.append(data)
            except:
                pass
                
    if not ablation_data:
        return
        
    # Group by stage and scenario
    for data in sorted(ablation_data, key=lambda x: (x.get('stage', ''), x.get('scenario', ''), x.get('level', 0))):
        stage = data.get('stage', 'unknown')
        scenario = data.get('scenario', 'unknown')
        level = data.get('level', 0)
        
        lines.append(f"Stage: {stage} | Scenario: {scenario} | Level: {level}")
        metrics = data.get('metrics', {})
        for method, m_data in metrics.items():
            if method.startswith('_'): continue
            f1 = m_data.get('f1', 0.0)
            lines.append(f"  - {method}: F1 = {f1:.3f}")
        lines.append("")
        
    with open(output_dir / 'ablation_summary.txt', 'w') as f:
        f.write('\n'.join(lines))
        
    print("  ✓ Generated: ablation_summary.txt")


def _generate_expert_summary(results: Dict, detections: Dict, output_dir: Path):
    """Generate expert summary interpretation"""
    
    lines = []
    lines.append("KNOWLEDGE GRAPH ANOMALY ANALYSIS REPORT")
    lines.append("=======================================")
    lines.append(f"Dataset: {list(results.values())[0]['dataset']}")
    lines.append("")
    
    # 1. Method Analysis
    lines.append("1. METHOD ANALYSIS")
    lines.append("------------------")
    
    display_names = {
        'ode': 'Neural ODE (Structural Consistency)',
        'TransE': 'PyKEEN-TransE (Translational Distance)',
        'MuRE': 'PyKEEN-MuRE (Euclidean Distance)',
        'RotatE': 'PyKEEN-RotatE (Rotational Distance)',
        'ComplEx': 'PyKEEN-ComplEx (Semantic Matching)',
        'DistMult': 'PyKEEN-DistMult (Bilinear Factorization)',
        'TuckER': 'PyKEEN-TuckER (Tucker Decomposition)'
    }
    
    for method, res in results.items():
        name = display_names.get(method, method)
        n_det = res['detections']['n_detected']
        mean_score = res['detections']['mean_score']
        
        lines.append(f"\n{name}:")
        lines.append(f"  - Anomalies Detected: {n_det}")
        lines.append(f"  - Confidence Score:   {mean_score:.3f}")
        
        # Interpretation per method
        if method == 'ode':
            if n_det > 0:
                lines.append(f"  - Interpretation: Detected {n_det} violations of logical constraints (transitivity, symmetry) or embedding instability.")
            else:
                lines.append("  - Interpretation: The graph structure appears logically consistent with respect to defined constraints.")
        else:
            if n_det > 0:
                lines.append(f"  - Interpretation: Identified {n_det} triples with low plausibility scores compared to learned patterns.")
            else:
                lines.append("  - Interpretation: No statistical outliers found according to this embedding model.")

    lines.append("")
    
    # 2. Global Summary
    lines.append("2. GLOBAL SUMMARY")
    lines.append("-----------------")
    
    # Calculate unique anomalies
    all_triples = set()
    for dets in detections.values():
        for d in dets:
            t = d['triple']
            all_triples.add((t['subject'], t['relation'], t['object']))
            
    lines.append(f"Total Unique Anomalies: {len(all_triples)}")
    
    # Find consensus
    triple_sets = {m: set((d['triple']['subject'], d['triple']['relation'], d['triple']['object']) for d in dets) 
                  for m, dets in detections.items()}
    
    if len(triple_sets) > 1:
        consensus = set.intersection(*triple_sets.values())
        lines.append(f"High-Confidence Anomalies (Consensus): {len(consensus)}")
    
    lines.append("")
    
    # 3. Expert Conclusions
    lines.append("3. EXPERT CONCLUSIONS")
    lines.append("---------------------")
    
    ode_count = results.get('ode', {}).get('detections', {}).get('n_detected', 0)
    pykeen_counts = [res['detections']['n_detected'] for m, res in results.items() if m != 'ode']
    avg_pykeen = np.mean(pykeen_counts) if pykeen_counts else 0
    
    lines.append("Based on the comparative analysis of geometric (ODE) and statistical (KGE) methods:")
    lines.append("")
    
    if len(all_triples) == 0:
        lines.append("• ROBUST DATASET: No anomalies were detected by any method. The dataset appears to be clean, consistent, and well-structured.")
    
    elif ode_count > avg_pykeen * 1.5:
        lines.append("• STRUCTURAL INCONSISTENCY DOMINANT: The Neural ODE method detected significantly more anomalies than statistical models.")
        lines.append("  This suggests the graph contains logical contradictions (e.g., violations of transitivity or hierarchy) that standard KGE models might overlook or overfit.")
        lines.append("  Recommendation: Review ontology definitions and hierarchical relations (subClassOf, partOf).")
        
    elif avg_pykeen > ode_count * 1.5:
        lines.append("• NOISY DISTRIBUTIONS: Statistical models detected more anomalies than the structural ODE approach.")
        lines.append("  This indicates the dataset likely contains random contradictions or 'facts' that don't fit the general statistical patterns, even if they don't violate strict logical rules.")
        lines.append("  Recommendation: Perform statistical outlier removal and check for data entry errors.")
        
    elif abs(ode_count - avg_pykeen) < max(ode_count, avg_pykeen) * 0.2:
        lines.append("• BALANCED DETECTION: Both structural and statistical methods detected a similar volume of anomalies.")
        lines.append("  This suggests a mix of structural issues and random noise.")
        lines.append("  Recommendation: Prioritize the 'Consensus' anomalies as they are likely definitive errors.")
        
    else:
        lines.append("• COMPLEMENTARY INSIGHTS: Methods show varied sensitivity to different types of errors.")
        lines.append("  Neural ODE is flagging continuous-time stability issues, while KGEs are flagging plausibility bounds.")
    
    lines.append("")
    
    # 4. Method Complementarity Analysis
    lines.append("4. METHOD COMPLEMENTARITY ANALYSIS")
    lines.append("----------------------------------")
    lines.append("")
    lines.append("Each detection method targets different error types:")
    lines.append("")
    lines.append("• Neural ODE: Detects STRUCTURAL violations (transitivity, symmetry, hierarchy).")
    lines.append("  Best for: Ontology consistency, logical rule enforcement.")
    lines.append("")
    lines.append("• TransE/MuRE: Detect GEOMETRIC anomalies (translation/distance in embedding space).")
    lines.append("  Best for: Relational pattern violations, entity-type mismatches.")
    lines.append("")
    lines.append("• DistMult: Detects FACTORIZATION anomalies (bilinear compatibility failures).")
    lines.append("  Best for: Symmetric relation errors, entity similarity violations.")
    lines.append("")
    lines.append("• TuckER: Detects TENSOR DECOMPOSITION anomalies (multi-way interaction failures).")
    lines.append("  Best for: Complex relation patterns, higher-order dependencies.")
    lines.append("")
    
    # Check if methods are truly complementary (low overlap)
    if len(triple_sets) > 1:
        overlaps = []
        for m1 in triple_sets:
            for m2 in triple_sets:
                if m1 != m2 and len(triple_sets[m1]) > 0:
                    overlap_pct = len(triple_sets[m1] & triple_sets[m2]) / len(triple_sets[m1]) * 100
                    overlaps.append(overlap_pct)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        
        if avg_overlap < 30:
            lines.append(f"COMPLEMENTARITY: HIGH (avg overlap: {avg_overlap:.1f}%)")
            lines.append("  Methods are detecting different sets of errors → Combined use recommended.")
        elif avg_overlap < 60:
            lines.append(f"COMPLEMENTARITY: MODERATE (avg overlap: {avg_overlap:.1f}%)")
            lines.append("  Some overlap, but each method still provides unique detections.")
        else:
            lines.append(f"COMPLEMENTARITY: LOW (avg overlap: {avg_overlap:.1f}%)")
            lines.append("  High redundancy between methods → Could simplify pipeline.")
    
    lines.append("")
    lines.append("Signed,")
    lines.append("AI Knowledge Graph Reliability Expert")
    
    # Save
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))
    
    print("  ✓ Generated: summary.txt")


def _generate_global_analysis(results: Dict, detections: Dict, output_dir: Path):
    """Generate global analysis summary"""
    
    lines = []
    lines.append("="*80)
    lines.append("COMPARATIVE ANALYSIS: NEURAL ODE VS PYKEEN METHODS")
    lines.append("="*80)
    lines.append("")
    
    dataset_name = list(results.values())[0]['dataset']
    n_triples = list(results.values())[0].get('n_triples', 'Unknown')
    
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Total triples: {n_triples}")
    lines.append("")
    
    # Method comparison
    lines.append("-"*80)
    lines.append("1. DETECTION PERFORMANCE")
    lines.append("-"*80)
    lines.append("")
    
    display_names = {
        'ode': 'Neural ODE',
        'TransE': 'PyKEEN-TransE',
        'RotatE': 'PyKEEN-RotatE',
        'ComplEx': 'PyKEEN-ComplEx'
    }
    
    for method in ['ode', 'TransE', 'RotatE', 'ComplEx']:
        if method in results:
            data = results[method]
            name = display_names.get(method, method)
            n_det = data['detections']['n_detected']
            mean_score = data['detections']['mean_score']
            exec_time = data['execution_time']
            
            lines.append(f"{name:20s}: {n_det:5d} anomalies | "
                        f"Avg score: {mean_score:.3f} | "
                        f"Time: {exec_time:.1f}s")
    
    lines.append("")
    
    # Key insights
    lines.append("-"*80)
    lines.append("2. KEY INSIGHTS")
    lines.append("-"*80)
    lines.append("")
    
    # Find best performer
    best_method = max(results.keys(), key=lambda m: results[m]['detections']['n_detected'])
    best_count = results[best_method]['detections']['n_detected']
    
    lines.append(f"• Best detector: {display_names.get(best_method, best_method)} ({best_count} anomalies)")
    
    # Find fastest
    fastest_method = min(results.keys(), key=lambda m: results[m]['execution_time'])
    fastest_time = results[fastest_method]['execution_time']
    
    lines.append(f"• Fastest method: {display_names.get(fastest_method, fastest_method)} ({fastest_time:.1f}s)")
    
    # Compare ODE vs PyKEEN
    if 'ode' in results:
        ode_count = results['ode']['detections']['n_detected']
        pykeen_counts = [results[m]['detections']['n_detected'] 
                        for m in ['TransE', 'RotatE', 'ComplEx'] if m in results]
        
        if pykeen_counts:
            avg_pykeen = np.mean(pykeen_counts)
            diff = ode_count - avg_pykeen
            
            if diff > 0:
                lines.append(f"• Neural ODE detected {diff:.0f} more anomalies than PyKEEN methods (on average)")
            else:
                lines.append(f"• PyKEEN methods detected {-diff:.0f} more anomalies than Neural ODE (on average)")
    
    lines.append("")
    
    # Save
    with open(output_dir / 'global_analysis.txt', 'w') as f:
        f.write('\n'.join(lines))
    
    print("  ✓ Generated: global_analysis.txt")


def _generate_overlap_analysis(detections: Dict, output_dir: Path):
    """Analyze overlap between methods"""
    
    # Build sets of detected triples
    triple_sets = {}
    for method, dets in detections.items():
        triple_set = set()
        for d in dets:
            t = d['triple']
            triple_set.add((t['subject'], t['relation'], t['object']))
        triple_sets[method] = triple_set
    
    # Compute overlaps
    analysis = {
        'total_unique': len(set.union(*triple_sets.values())) if triple_sets else 0,
        'method_counts': {m: len(s) for m, s in triple_sets.items()},
        'overlaps': {}
    }
    
    methods = list(triple_sets.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            overlap = len(triple_sets[m1] & triple_sets[m2])
            analysis['overlaps'][f'{m1}_vs_{m2}'] = {
                'count': overlap,
                'percentage_of_m1': (overlap / len(triple_sets[m1]) * 100) if triple_sets[m1] else 0,
                'percentage_of_m2': (overlap / len(triple_sets[m2]) * 100) if triple_sets[m2] else 0
            }
    
    # Find consensus anomalies (detected by all methods)
    if len(triple_sets) > 1:
        consensus = set.intersection(*triple_sets.values())
        analysis['consensus_anomalies'] = {
            'count': len(consensus),
            'triples': [{'subject': s, 'relation': r, 'object': o} for s, r, o in list(consensus)[:20]]
        }
    
    # Save as JSON
    with open(output_dir / 'overlap_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("  ✓ Generated: overlap_analysis.json")


def _generate_detailed_report(results: Dict, detections: Dict, output_dir: Path):
    """Generate detailed comparison report"""
    
    report = {
        'summary': {},
        'methods': {},
        'top_anomalies_by_method': {}
    }
    
    # Summary statistics
    report['summary']['n_methods'] = len(results)
    report['summary']['total_detections'] = sum(r['detections']['n_detected'] for r in results.values())
    
    # Per-method details
    for method, res in results.items():
        report['methods'][method] = {
            'full_name': res['method'],
            'detections': res['detections']['n_detected'],
            'mean_score': res['detections']['mean_score'],
            'execution_time': res['execution_time'],
            'parameters': res.get('parameters', {})
        }
        
        # Top 10 anomalies
        if method in detections:
            sorted_dets = sorted(detections[method], key=lambda x: x['score'], reverse=True)
            report['top_anomalies_by_method'][method] = []
            
            for d in sorted_dets[:10]:
                item = d.copy()
                # Ensure expert judgments are included if available
                # (They should be in 'd' already if the upstream flow passed them)
                report['top_anomalies_by_method'][method].append(item)
    
    # Save
    with open(output_dir / 'detailed_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("  ✓ Generated: detailed_report.json")
