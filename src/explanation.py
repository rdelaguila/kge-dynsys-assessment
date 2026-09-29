"""
Explanation Module for KG Anomalies
===================================
Implements Chain-of-Thought (CoT) and QualIT strategies for explaining 
knowledge graph constraint violations and suggesting corrections.

Based on the strategy:
1. Key phrase extraction
2. Hallucination verification
3. Chain-of-thought explanation generation
4. Structured evaluation
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try importing transformers, handle if missing
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ChainOfThoughtExplainer:
    """
    Generates explanations for KG anomalies using CoT prompting.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = None):
        self.model_name = model_name
        self.pipeline = None
        
        # Default to True now that we use a lighter model
        enable_llm = os.environ.get("ENABLE_LLM", "True").lower() == "true"
        
        if TRANSFORMERS_AVAILABLE and enable_llm:
            try:
                self.device = 'mps' #device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Loading explanation model: {model_name} on {self.device}...")
                
                try:
                    # Try local load first
                    print("Attempting to load model from local cache...")
                    cache_dir = os.path.join(os.getcwd(), ".model_cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).to(self.device)
                    print("Loaded from local cache.")
                except Exception:
                    # Fallback to download and save to cache
                    print("Local model not found. Downloading to cache...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
                
                self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
                print(f"Model {model_name} loaded successfully.")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Falling back to mock responses.")
        else:
            if not enable_llm:
                print("LLM explanations disabled (ENABLE_LLM=False). Using mock responses.")
            else:
                print("Transformers not available. Explanations will be template-based.")

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Wrapper for generation"""
        if self.pipeline:
            try:
                output = self.pipeline(prompt, max_new_tokens=max_new_tokens, temperature=0.2)
                return output[0]['generated_text']
            except:
                return ""
        else:
            # Mock response for demo purposes
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses based on prompt content"""
        if "VERIFICATION TASK" in prompt:
            return "VERIFICATION RESULT (True/False): True"
        elif "Generate a comprehensive explanation" in prompt or "Explain why the following triple" in prompt:
            return json.dumps({
                "explanation": "The triple violates the domain constraint because the subject is not of the expected type.",
                "coherence": 4,
                "plausibility_analysis": "High energy confirms the logical violation.",
                "risk_level": "Confirmed Error",
                "reasoning": "Clear violation of defined schema."
            })
        elif "Suggest 3 valid alternatives" in prompt:
            return json.dumps({
                "suggestions": ["Alternative 1", "Alternative 2"],
                "reasoning": "These entities fit the domain."
            })
        return ""

    def explain_violation(self, triple, violation_type: str, constraint_info: str, 
                          detection_method: str = "unknown", detection_score: float = 0.0) -> Dict:
        """
        Generate CoT explanation using a Two-Step Expert System:
        1. Semantic Domain Expert (Analyzes facts + detection context)
        2. Analytical Judge (Renders verdict based on expert analysis)
        
        Args:
            triple: The anomalous triple
            violation_type: Type of violation detected
            constraint_info: Additional constraint information
            detection_method: Which method detected this (ode, TransE, MuRE, DistMult)
            detection_score: The anomaly score from the detector
        """
        
        # Define method descriptions for context
        method_descriptions = {
            "ode": "Neural ODE: Models embeddings as a dynamical system. High scores indicate the triple destabilizes the system's equilibrium, suggesting structural inconsistency with logical constraints (transitivity, symmetry).",
            "TransE": "TransE: Translational embedding model where h + r ≈ t. High scores mean the translation distance is abnormally large, indicating the triple doesn't fit learned relational patterns.",
            "MuRE": "MuRE: Multi-relational Euclidean model using relation-specific transformations. High scores indicate the transformed embedding distance exceeds normal bounds.",
            "DistMult": "DistMult: Bilinear model using element-wise products. High scores suggest the triple's factorization doesn't match the learned entity-relation compatibilities.",
            "TuckER": "TuckER: Tucker tensor decomposition model with a learned core tensor. High scores indicate the triple doesn't fit the multi-way interaction pattern between entities and relations captured by the factorization.",
            "unknown": "Unknown detection method."
        }
        
        method_desc = method_descriptions.get(detection_method, method_descriptions["unknown"])
        
        # --- STAGE 1: SEMANTIC DOMAIN EXPERT ---
        prompt_expert = f"""You are a World-Class Knowledge Graph and Semantic Web Expert.

CONTEXT: An automated anomaly detection system has flagged the following triple.

DETECTION METHOD USED:
{method_desc}

FLAGGED TRIPLE: ({triple.subject}, {triple.relation}, {triple.object})
DETECTED ISSUE: {violation_type}
DETECTION SCORE: {detection_score:.4f} (higher = more anomalous)
ADDITIONAL INFO: {constraint_info}

YOUR TASK: Analyze this potential violation from 4 different perspectives. Be specific to THIS triple.

1. JUDGMENT A (Standard Violation): 
   Assuming the detection is correct, explain WHY this triple is semantically or logically wrong.

2. JUDGMENT B (Data Quality Issue): 
   Could this be a typo, OCR error, or extraction mistake? What would the correct triple look like?

3. JUDGMENT C (Polysemy/Context Ambiguity): 
   Could '{triple.subject}' or '{triple.object}' have multiple meanings? Is there a context where this is valid?

4. JUDGMENT D (Rare but Valid Exception): 
   Is there a documented real-world exception where this triple holds true despite seeming wrong?

Output your 4 judgments as a numbered list. Be concise but specific.
"""
        expert_analysis = self._generate(prompt_expert, max_new_tokens=400)

        # --- STAGE 2: ANALYTICAL JUDGE ---
        prompt_judge = f"""You are the Chief Analytical Judge for Knowledge Graph Quality Assurance.

You must render a FINAL VERDICT on a flagged anomaly based on the Domain Expert's analysis.

TRIPLE UNDER REVIEW: ({triple.subject}, {triple.relation}, {triple.object})
VIOLATION TYPE: {violation_type}
DETECTION METHOD: {detection_method} (Score: {detection_score:.4f})

EXPERT'S 4 JUDGMENTS:
{expert_analysis}

CHAIN OF THOUGHT VERDICT PROCESS:
1. Which judgment (A, B, C, or D) is most plausible given the evidence?
2. Does the detection score ({detection_score:.4f}) support or contradict this judgment?
3. If the score is HIGH (>0.7), favor Judgment A unless C or D provide strong counter-evidence.
4. If the score is MEDIUM (0.3-0.7), consider B or C carefully.
5. If the score is LOW (<0.3), this may be a false positive.

FINAL DECISION (generate valid JSON):
{{
    "chosen_judgment": "A/B/C/D",
    "explanation": "Synthesized explanation for this specific triple",
    "coherence": 4,
    "plausibility_analysis": "Why you chose this judgment over the others",
    "risk_level": "Confirmed Error / Likely Error / Needs Verification / Likely False Positive",
    "recommended_action": "delete / correct / verify / keep",
    "reasoning": "Step-by-step logic for your decision"
}}
"""
        response_text = self._generate(prompt_judge, max_new_tokens=400)
        result = self._parse_json_response(response_text)
        # Attach metadata for full traceability
        result["expert_judgments"] = expert_analysis
        result["detection_method"] = detection_method
        result["detection_score"] = detection_score
        return result

    def suggest_corrections(self, triple, violation_type: str, context_candidates: List[str]) -> Dict:
        """
        Suggest corrections using CoT.
        """
        candidates_str = ", ".join(context_candidates[:5])
        
        prompt = f"""You are an expert in Knowledge Graph repair.

TASK: Suggest corrections for the invalid triple, ONLY if a clear fix exists.

TRIPLE: ({triple.subject}, {triple.relation}, {triple.object})
ERROR: {violation_type}
CANDIDATES FROM EMBEDDINGS: {candidates_str}

CHAIN OF THOUGHT STEPS:
1. Identify which part of the triple is likely incorrect (Subject, Relation, or Object).
2. Evaluate the provided candidates for semantic fit.
3. If no candidate makes semantic sense, suggest "VERIFY" instead of a correction.

Generate a JSON response:
{{
    "suggestions": [
        {{"action": "replace_object", "value": "candidate1", "reason": "Fits domain"}},
        {{"action": "verify_triple", "value": "N/A", "reason": "No clear correction found"}}
    ],
    "reasoning": "Why these corrections work"
}}
"""
        response_text = self._generate(prompt, max_new_tokens=300)
        return self._parse_json_response(response_text)

    def generate_global_explanation(self, anomaly_reports: List, max_samples: int = 20) -> Dict:
        """
        Generate a global explanation by analyzing patterns across multiple anomalies.
        
        Args:
            anomaly_reports: List of anomaly reports to analyze
            max_samples: Maximum number of anomalies to sample for analysis
            
        Returns:
            Dict with global analysis including patterns, common issues, and recommendations
        """
        # Sample top anomalies
        sampled_reports = sorted(anomaly_reports, key=lambda r: r.anomaly_score, reverse=True)[:max_samples]
        
        # Group anomalies by type
        type_groups = {}
        for report in sampled_reports:
            for anom_type in report.anomaly_types:
                if anom_type not in type_groups:
                    type_groups[anom_type] = []
                type_groups[anom_type].append(report)
        
        # Prepare summary for LLM
        summary_lines = []
        summary_lines.append(f"Total anomalies analyzed: {len(sampled_reports)}")
        summary_lines.append(f"Anomaly type distribution:")
        for anom_type, reports in type_groups.items():
            summary_lines.append(f"  - {anom_type}: {len(reports)} instances")
        
        # Sample representative examples
        examples = []
        for anom_type, reports in list(type_groups.items())[:5]:  # Top 5 types
            if reports:
                example = reports[0]  # Take first example
                examples.append(f"  Type '{anom_type}': ({example.triple.subject}, {example.triple.relation}, {example.triple.object}) [score: {example.anomaly_score:.3f}]")
        
        summary_text = "\n".join(summary_lines)
        examples_text = "\n".join(examples)
        
        # Generate global explanation
        prompt = f"""You are an expert in Knowledge Graph quality analysis.

TASK: Analyze the following anomaly detection results and provide a GLOBAL explanation of the main issues found in the knowledge graph.

SUMMARY OF DETECTED ANOMALIES:
{summary_text}

REPRESENTATIVE EXAMPLES:
{examples_text}

CHAIN OF THOUGHT ANALYSIS:
1. Identify the main patterns and commonalities across the detected anomalies.
2. Determine if there are systematic issues (e.g., data quality problems, schema violations, inconsistencies).
3. Assess the severity and impact of these issues on the knowledge graph integrity.
4. Provide actionable recommendations for addressing the root causes.

Generate a JSON response:
{{
    "global_summary": "High-level summary of the main issues detected",
    "key_patterns": ["Pattern 1", "Pattern 2", "Pattern 3"],
    "severity_assessment": "low/medium/high",
    "root_causes": ["Potential cause 1", "Potential cause 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "affected_relations": ["relation1", "relation2"],
    "estimated_impact": "Brief description of the impact on the KG"
}}
"""
        
        response_text = self._generate(prompt, max_new_tokens=500)
        global_analysis = self._parse_json_response(response_text)
        
        # Add statistics
        global_analysis["statistics"] = {
            "total_anomalies": len(anomaly_reports),
            "sampled_anomalies": len(sampled_reports),
            "unique_types": len(type_groups),
            "type_distribution": {k: len(v) for k, v in type_groups.items()}
        }
        
        return global_analysis

    def _parse_json_response(self, text: str) -> Dict:
        """Extract and parse JSON from text"""
        try:
            # Simple extraction logic
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback
        return {
            "explanation": "Could not generate detailed explanation.",
            "suggestions": [],
            "reasoning": "Parsing error"
        }

