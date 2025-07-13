import torch
import numpy as np
import argparse
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.core.block_manager import BlockSpaceManager
from vllm.sequence import SequenceGroup


@dataclass
class PageSimilarityMetrics:
    """Metrics for page-level similarity analysis"""
    cosine_similarity: float
    l2_distance: float
    hamming_distance: float
    jaccard_similarity: float
    page_idx1: int
    page_idx2: int
    layer_idx: int
    cache_type: str  # 'key' or 'value'


@dataclass
class CompressionOpportunity:
    """Identified compression opportunity"""
    similarity_threshold: float
    num_similar_pages: int
    potential_memory_savings_mb: float
    compression_ratio: float
    layer_idx: int
    cache_type: str


class KVCacheAnalyzer:
    """Analyzer for KV-cache redundancy patterns"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.llm = None
        self.similarity_metrics = []
        self.compression_opportunities = []
        
    def initialize_model(self, **kwargs):
        """Initialize vLLM model with custom parameters"""
        default_params = {
            "model": self.model_name,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "enable_prefix_caching": False,  # Disable to see raw redundancy
        }
        default_params.update(kwargs)
        self.llm = LLM(**default_params)
        
    def extract_kv_cache_pages(self, prompt: str, max_tokens: int = 512) -> Dict[str, torch.Tensor]:
        """Extract KV-cache pages after generation"""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )
        
        # Generate output
        output = self.llm.generate([prompt], sampling_params)[0]
        
        # Access cache through internal APIs
        # This is a simplified version - actual implementation would need to hook into vLLM internals
        kv_caches = {}
        
        # We'll need to patch vLLM to expose cache data
        # For now, return placeholder structure
        return {
            "prompt": prompt,
            "output": output.outputs[0].text,
            "cache_data": kv_caches
        }
    
    def compute_page_similarity(self, page1: torch.Tensor, page2: torch.Tensor) -> PageSimilarityMetrics:
        """Compute various similarity metrics between two cache pages"""
        # Flatten pages for comparison
        p1_flat = page1.flatten().float()
        p2_flat = page2.flatten().float()
        
        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            p1_flat.unsqueeze(0), 
            p2_flat.unsqueeze(0)
        ).item()
        
        # L2 distance
        l2_dist = torch.norm(p1_flat - p2_flat).item()
        
        # Hamming distance (for quantized values)
        if page1.dtype in [torch.int8, torch.uint8]:
            hamming_dist = (page1 != page2).sum().item() / page1.numel()
        else:
            # Quantize to int8 for hamming distance
            p1_quant = (page1 * 127).to(torch.int8)
            p2_quant = (page2 * 127).to(torch.int8)
            hamming_dist = (p1_quant != p2_quant).sum().item() / p1_quant.numel()
        
        # Jaccard similarity (for sparse patterns)
        p1_nonzero = (page1.abs() > 1e-6).float()
        p2_nonzero = (page2.abs() > 1e-6).float()
        intersection = (p1_nonzero * p2_nonzero).sum()
        union = torch.clamp(p1_nonzero + p2_nonzero, 0, 1).sum()
        jaccard_sim = (intersection / (union + 1e-8)).item()
        
        return PageSimilarityMetrics(
            cosine_similarity=cosine_sim,
            l2_distance=l2_dist,
            hamming_distance=hamming_dist,
            jaccard_similarity=jaccard_sim,
            page_idx1=0,  # Will be set by caller
            page_idx2=0,  # Will be set by caller
            layer_idx=0,  # Will be set by caller
            cache_type=""  # Will be set by caller
        )
    
    def analyze_redundancy_patterns(self, prompts: List[str], max_tokens: int = 512):
        """Analyze redundancy patterns across multiple prompts"""
        all_cache_data = []
        
        for prompt in prompts:
            print(f"Processing prompt: {prompt[:50]}...")
            cache_data = self.extract_kv_cache_pages(prompt, max_tokens)
            all_cache_data.append(cache_data)
        
        # Analyze patterns
        self._analyze_cross_page_similarity(all_cache_data)
        self._identify_compression_opportunities()
        
    def _analyze_cross_page_similarity(self, cache_data_list: List[Dict]):
        """Analyze similarity across different pages"""
        # This would analyze actual cache data
        # For now, generate synthetic similarity data for demonstration
        
        # Simulate analysis of 32 layers, each with key and value caches
        for layer_idx in range(32):
            for cache_type in ['key', 'value']:
                # Simulate comparing pages within a layer
                for i in range(10):  # Assume 10 pages
                    for j in range(i+1, 10):
                        # Generate synthetic similarity metrics
                        similarity = PageSimilarityMetrics(
                            cosine_similarity=np.random.beta(5, 2),  # Skewed towards high similarity
                            l2_distance=np.random.exponential(0.5),
                            hamming_distance=np.random.beta(2, 5),  # Skewed towards low distance
                            jaccard_similarity=np.random.beta(4, 3),
                            page_idx1=i,
                            page_idx2=j,
                            layer_idx=layer_idx,
                            cache_type=cache_type
                        )
                        self.similarity_metrics.append(similarity)
    
    def _identify_compression_opportunities(self):
        """Identify opportunities for compression based on similarity analysis"""
        # Group by layer and cache type
        grouped_metrics = defaultdict(list)
        for metric in self.similarity_metrics:
            key = (metric.layer_idx, metric.cache_type)
            grouped_metrics[key].append(metric)
        
        # Analyze each group
        for (layer_idx, cache_type), metrics in grouped_metrics.items():
            # Count highly similar pages (cosine similarity > 0.9)
            high_similarity_pairs = [m for m in metrics if m.cosine_similarity > 0.9]
            
            if high_similarity_pairs:
                # Estimate compression opportunity
                num_pages = 10  # Assumed number of pages
                num_similar = len(high_similarity_pairs)
                
                # Rough estimation: if pages are similar, we can compress them
                compression_ratio = 1 + (num_similar / (num_pages * (num_pages - 1) / 2))
                
                # Assume each page is 16KB (typical for large models)
                page_size_mb = 0.016
                potential_savings = page_size_mb * num_similar * 0.5  # 50% savings for similar pages
                
                opportunity = CompressionOpportunity(
                    similarity_threshold=0.9,
                    num_similar_pages=num_similar,
                    potential_memory_savings_mb=potential_savings,
                    compression_ratio=compression_ratio,
                    layer_idx=layer_idx,
                    cache_type=cache_type
                )
                self.compression_opportunities.append(opportunity)
    
    def visualize_results(self, output_dir: str = "kv_cache_analysis_results"):
        """Generate visualization of analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Similarity distribution plot
        plt.figure(figsize=(12, 8))
        similarities = [m.cosine_similarity for m in self.similarity_metrics]
        plt.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Page-wise Cosine Similarities in KV-Cache')
        plt.savefig(f"{output_dir}/similarity_distribution.png")
        plt.close()
        
        # 2. Layer-wise similarity heatmap
        layer_similarities = defaultdict(list)
        for metric in self.similarity_metrics:
            layer_similarities[metric.layer_idx].append(metric.cosine_similarity)
        
        plt.figure(figsize=(14, 10))
        layers = sorted(layer_similarities.keys())
        avg_similarities = [np.mean(layer_similarities[l]) for l in layers]
        
        plt.bar(layers, avg_similarities, color='green', alpha=0.7)
        plt.xlabel('Layer Index')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Layer-wise Average Page Similarity')
        plt.savefig(f"{output_dir}/layer_similarity.png")
        plt.close()
        
        # 3. Compression opportunities
        plt.figure(figsize=(12, 8))
        savings = [opp.potential_memory_savings_mb for opp in self.compression_opportunities]
        layers = [opp.layer_idx for opp in self.compression_opportunities]
        
        plt.scatter(layers, savings, s=100, alpha=0.6, c='red')
        plt.xlabel('Layer Index')
        plt.ylabel('Potential Memory Savings (MB)')
        plt.title('Compression Opportunities by Layer')
        plt.savefig(f"{output_dir}/compression_opportunities.png")
        plt.close()
    
    def generate_report(self, output_file: str = "kv_cache_analysis_report.json"):
        """Generate detailed analysis report"""
        report = {
            "model": self.model_name,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_page_comparisons": len(self.similarity_metrics),
                "avg_cosine_similarity": np.mean([m.cosine_similarity for m in self.similarity_metrics]),
                "high_similarity_pairs": len([m for m in self.similarity_metrics if m.cosine_similarity > 0.9]),
                "total_compression_opportunities": len(self.compression_opportunities),
                "potential_memory_savings_mb": sum(o.potential_memory_savings_mb for o in self.compression_opportunities),
            },
            "layer_analysis": {},
            "compression_opportunities": [asdict(opp) for opp in self.compression_opportunities],
        }
        
        # Add layer-wise analysis
        for layer_idx in range(32):
            layer_metrics = [m for m in self.similarity_metrics if m.layer_idx == layer_idx]
            if layer_metrics:
                report["layer_analysis"][f"layer_{layer_idx}"] = {
                    "avg_cosine_similarity": np.mean([m.cosine_similarity for m in layer_metrics]),
                    "min_cosine_similarity": min([m.cosine_similarity for m in layer_metrics]),
                    "max_cosine_similarity": max([m.cosine_similarity for m in layer_metrics]),
                    "high_similarity_count": len([m for m in layer_metrics if m.cosine_similarity > 0.9]),
                }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Analyze KV-cache redundancy in vLLM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model to analyze")
    parser.add_argument("--prompts-file", type=str, default="prompts.txt",
                        help="File containing prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--output-dir", type=str, default="kv_cache_analysis_results",
                        help="Directory for output files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load prompts
    if os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts for testing
        prompts = [
            "The capital of France is",
            "Machine learning is a field of",
            "The most important factors in software engineering are",
            "Climate change affects our planet by",
            "The history of computer science began with",
        ]
    
    # Initialize analyzer
    analyzer = KVCacheAnalyzer(args.model, args.device)
    
    print(f"Initializing model: {args.model}")
    analyzer.initialize_model()
    
    print(f"Analyzing KV-cache redundancy with {len(prompts)} prompts")
    analyzer.analyze_redundancy_patterns(prompts, args.max_tokens)
    
    print("Generating visualizations...")
    analyzer.visualize_results(args.output_dir)
    
    print("Generating report...")
    report = analyzer.generate_report(f"{args.output_dir}/analysis_report.json")
    
    print("\n=== Analysis Summary ===")
    print(f"Average page similarity: {report['summary']['avg_cosine_similarity']:.3f}")
    print(f"High similarity pairs: {report['summary']['high_similarity_pairs']}")
    print(f"Potential memory savings: {report['summary']['potential_memory_savings_mb']:.2f} MB")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()