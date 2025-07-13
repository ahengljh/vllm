#!/usr/bin/env python3
"""
Main script to run KV-cache redundancy experiments
This script coordinates different experiments to measure page-level redundancy in KV-cache
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any
import torch
import numpy as np
from dataclasses import dataclass, asdict

# Import the monitoring module first to install hooks
import kv_cache_page_monitor

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    model: str
    prompts: List[str]
    max_tokens: int
    temperature: float
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048


class KVCacheExperimentRunner:
    """Runner for KV-cache experiments"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.name}")
        print(f"Model: {config.model}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Clear previous captures
        kv_cache_page_monitor.cache_captures.clear()
        
        # Initialize model
        llm = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enable_prefix_caching=False,  # Disable to see raw patterns
        )
        
        # Run inference on all prompts
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        outputs = []
        for i, prompt in enumerate(config.prompts):
            print(f"Processing prompt {i+1}/{len(config.prompts)}: {prompt[:50]}...")
            output = llm.generate([prompt], sampling_params)
            outputs.append(output[0])
            
            # Small delay to allow monitoring to capture data
            time.sleep(0.1)
        
        # Get analysis results
        analysis = kv_cache_page_monitor.analyze_cache_patterns()
        
        # Compute statistics
        experiment_results = {
            "config": asdict(config),
            "timing": {
                "start_time": start_time,
                "end_time": time.time(),
                "duration": time.time() - start_time,
            },
            "outputs": [
                {
                    "prompt": prompt,
                    "generated_text": output.outputs[0].text,
                    "num_tokens": len(output.outputs[0].token_ids),
                }
                for prompt, output in zip(config.prompts, outputs)
            ],
            "cache_analysis": analysis,
            "memory_stats": self._get_memory_stats(),
        }
        
        # Save individual experiment results
        exp_file = os.path.join(self.output_dir, f"{config.name}_results.json")
        with open(exp_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"Experiment completed in {experiment_results['timing']['duration']:.2f}s")
        self._print_analysis_summary(analysis)
        
        return experiment_results
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated()) / 1e9,
            }
        return {}
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print summary of analysis results"""
        print("\n--- Cache Analysis Summary ---")
        
        for cache_type in ['key', 'value']:
            if cache_type in analysis['similarity_analysis']:
                stats = analysis['similarity_analysis'][cache_type]
                print(f"\n{cache_type.upper()} Cache:")
                print(f"  Mean similarity: {stats['mean_similarity']:.3f}")
                print(f"  Max similarity: {stats['max_similarity']:.3f}")
                print(f"  High similarity ratio: {stats['high_similarity_ratio']:.3f}")
                
                if cache_type in analysis['redundancy_stats']:
                    redundancy = analysis['redundancy_stats'][cache_type]
                    print(f"  Redundancy ratio: {redundancy['redundancy_ratio']:.3f}")
                    print(f"  Potential memory savings: {redundancy['estimated_memory_savings_percent']:.1f}%")
    
    def run_all_experiments(self, experiments: List[ExperimentConfig]):
        """Run all experiments and generate final report"""
        all_results = []
        
        for exp_config in experiments:
            try:
                result = self.run_experiment(exp_config)
                all_results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"Error in experiment {exp_config.name}: {e}")
                continue
        
        # Generate final report
        self.generate_final_report()
        
        return all_results
    
    def generate_final_report(self):
        """Generate comprehensive report across all experiments"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_experiments": len(self.results),
            "experiments": [],
            "aggregate_stats": {},
        }
        
        # Collect stats across experiments
        all_similarities = {'key': [], 'value': []}
        all_redundancies = {'key': [], 'value': []}
        
        for result in self.results:
            exp_summary = {
                "name": result['config']['name'],
                "model": result['config']['model'],
                "duration": result['timing']['duration'],
            }
            
            # Extract key metrics
            for cache_type in ['key', 'value']:
                if cache_type in result['cache_analysis']['similarity_analysis']:
                    stats = result['cache_analysis']['similarity_analysis'][cache_type]
                    exp_summary[f"{cache_type}_mean_similarity"] = stats['mean_similarity']
                    all_similarities[cache_type].append(stats['mean_similarity'])
                    
                if cache_type in result['cache_analysis']['redundancy_stats']:
                    redundancy = result['cache_analysis']['redundancy_stats'][cache_type]
                    exp_summary[f"{cache_type}_redundancy"] = redundancy['redundancy_ratio']
                    all_redundancies[cache_type].append(redundancy['redundancy_ratio'])
            
            report['experiments'].append(exp_summary)
        
        # Compute aggregate statistics
        for cache_type in ['key', 'value']:
            if all_similarities[cache_type]:
                report['aggregate_stats'][f"{cache_type}_similarity"] = {
                    "mean": np.mean(all_similarities[cache_type]),
                    "std": np.std(all_similarities[cache_type]),
                    "min": np.min(all_similarities[cache_type]),
                    "max": np.max(all_similarities[cache_type]),
                }
            
            if all_redundancies[cache_type]:
                report['aggregate_stats'][f"{cache_type}_redundancy"] = {
                    "mean": np.mean(all_redundancies[cache_type]),
                    "std": np.std(all_redundancies[cache_type]),
                    "min": np.min(all_redundancies[cache_type]),
                    "max": np.max(all_redundancies[cache_type]),
                }
        
        # Save final report
        report_file = os.path.join(self.output_dir, "final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        print(f"Total experiments: {report['num_experiments']}")
        
        for cache_type in ['key', 'value']:
            if f"{cache_type}_similarity" in report['aggregate_stats']:
                stats = report['aggregate_stats'][f"{cache_type}_similarity"]
                print(f"\n{cache_type.upper()} Cache Similarity:")
                print(f"  Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        print(f"\nFull report saved to: {report_file}")


def create_default_experiments() -> List[ExperimentConfig]:
    """Create default experiment configurations"""
    
    # Common prompts for testing redundancy
    base_prompts = [
        "Write a detailed explanation of machine learning fundamentals including supervised and unsupervised learning.",
        "Explain the history and evolution of artificial intelligence from the 1950s to present day.",
        "Describe the architecture and working principles of transformer models in deep learning.",
        "What are the key differences between traditional programming and machine learning approaches?",
        "Provide a comprehensive overview of natural language processing techniques and applications.",
    ]
    
    # Prompts with repetitive patterns (to test redundancy)
    repetitive_prompts = [
        "List the numbers from 1 to 100, explaining the mathematical properties of each.",
        "Describe each planet in our solar system in detail, including size, composition, and characteristics.",
        "Explain each layer of the OSI networking model with examples and use cases.",
        "Write about each month of the year, including holidays, weather patterns, and cultural significance.",
        "Describe the characteristics of each element in the periodic table from Hydrogen to Iron.",
    ]
    
    # Code generation prompts (often have patterns)
    code_prompts = [
        "Implement a binary search tree in Python with insert, delete, and search operations.",
        "Write a React component for a todo list application with add, remove, and toggle functionality.",
        "Create a REST API in Flask with CRUD operations for a user management system.",
        "Implement quicksort algorithm in multiple programming languages: Python, Java, and C++.",
        "Write a SQL schema and queries for an e-commerce database with products, users, and orders.",
    ]
    
    experiments = [
        # Experiment 1: General knowledge (baseline)
        ExperimentConfig(
            name="general_knowledge",
            model="meta-llama/Llama-2-7b-hf",
            prompts=base_prompts,
            max_tokens=512,
            temperature=0.0,
        ),
        
        # Experiment 2: Repetitive content
        ExperimentConfig(
            name="repetitive_content",
            model="meta-llama/Llama-2-7b-hf",
            prompts=repetitive_prompts,
            max_tokens=1024,
            temperature=0.0,
        ),
        
        # Experiment 3: Code generation
        ExperimentConfig(
            name="code_generation",
            model="meta-llama/Llama-2-7b-hf",
            prompts=code_prompts,
            max_tokens=512,
            temperature=0.0,
        ),
        
        # Experiment 4: High temperature (more randomness)
        ExperimentConfig(
            name="high_temperature",
            model="meta-llama/Llama-2-7b-hf",
            prompts=base_prompts,
            max_tokens=512,
            temperature=1.0,
        ),
        
        # Experiment 5: Long context
        ExperimentConfig(
            name="long_context",
            model="meta-llama/Llama-2-7b-hf",
            prompts=["Tell me a very long and detailed story about " + topic for topic in 
                    ["space exploration", "ancient civilizations", "future technology"]],
            max_tokens=2048,
            temperature=0.7,
            max_model_len=4096,
        ),
    ]
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Run KV-cache redundancy experiments")
    parser.add_argument("--output-dir", type=str, default="experiment_results",
                        help="Directory to save results")
    parser.add_argument("--experiments", type=str, default=None,
                        help="JSON file with experiment configurations")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model for all experiments")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a quick test with minimal prompts")
    
    args = parser.parse_args()
    
    # Load or create experiments
    if args.experiments:
        with open(args.experiments, 'r') as f:
            exp_data = json.load(f)
            experiments = [ExperimentConfig(**exp) for exp in exp_data]
    else:
        experiments = create_default_experiments()
    
    # Override model if specified
    if args.model:
        for exp in experiments:
            exp.model = args.model
    
    # Quick test mode
    if args.quick_test:
        experiments = [experiments[0]]  # Only run first experiment
        experiments[0].prompts = experiments[0].prompts[:2]  # Only 2 prompts
        experiments[0].max_tokens = 128  # Shorter generation
    
    # Run experiments
    runner = KVCacheExperimentRunner(args.output_dir)
    runner.run_all_experiments(experiments)
    
    # Save monitoring data
    kv_cache_page_monitor.save_analysis_results(
        os.path.join(args.output_dir, "monitoring_data")
    )


if __name__ == "__main__":
    main()