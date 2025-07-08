"""
Performance Profiler for Financial Risk Analysis
===============================================

This module provides comprehensive performance monitoring and profiling
capabilities to measure the impact of optimizations in the financial
risk analysis codebase.

Features:
- Function-level execution time tracking
- Memory usage monitoring
- Cache performance analysis
- Before/after comparison tools
- Automated benchmark reporting
"""

import time
import psutil
import functools
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
import gc
import tracemalloc
from pathlib import Path
import json

class PerformanceProfiler:
    """
    Comprehensive performance profiler for financial analysis functions
    
    Features:
    - Automatic function timing
    - Memory usage tracking
    - Performance comparison
    - Report generation
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.metrics = {}
        self.memory_snapshots = {}
        self.enable_memory_tracking = enable_memory_tracking
        self.logger = logging.getLogger(__name__)
        
        # Initialize process monitoring
        self.process = psutil.Process()
        
        if enable_memory_tracking:
            tracemalloc.start()
    
    def profile_function(self, func_name: Optional[str] = None, track_memory: bool = True):
        """
        Decorator to profile function execution
        
        Args:
            func_name: Custom name for the function (uses __name__ if None)
            track_memory: Whether to track detailed memory usage
            
        Returns:
            Decorated function with profiling capabilities
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                
                # Pre-execution metrics
                start_time = time.time()
                start_cpu = time.process_time()
                
                if track_memory and self.enable_memory_tracking:
                    gc.collect()  # Force garbage collection for accurate measurement
                    start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    tracemalloc_start = tracemalloc.get_traced_memory()[0]
                else:
                    start_memory = 0
                    tracemalloc_start = 0
                
                # Execute function
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error_msg = None
                except Exception as e:
                    result = None
                    success = False
                    error_msg = str(e)
                    self.logger.error(f"Function {name} failed: {e}")
                
                # Post-execution metrics
                end_time = time.time()
                end_cpu = time.process_time()
                
                if track_memory and self.enable_memory_tracking:
                    end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    tracemalloc_end = tracemalloc.get_traced_memory()[0]
                    memory_delta = end_memory - start_memory
                    tracemalloc_delta = (tracemalloc_end - tracemalloc_start) / 1024 / 1024  # MB
                else:
                    memory_delta = 0
                    tracemalloc_delta = 0
                
                # Store metrics
                self.metrics[name] = {
                    'wall_time': end_time - start_time,
                    'cpu_time': end_cpu - start_cpu,
                    'memory_delta': memory_delta,
                    'tracemalloc_delta': tracemalloc_delta,
                    'timestamp': datetime.now(),
                    'success': success,
                    'error_msg': error_msg,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                if success:
                    return result
                else:
                    raise Exception(error_msg)
                    
            return wrapper
        return decorator
    
    @contextmanager
    def profile_context(self, context_name: str):
        """
        Context manager for profiling code blocks
        
        Args:
            context_name: Name identifier for the profiled context
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            self.metrics[context_name] = {
                'wall_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': datetime.now(),
                'success': True,
                'context_type': 'block'
            }
    
    def benchmark_function(
        self, 
        func: Callable, 
        *args, 
        runs: int = 5, 
        warmup: int = 1,
        **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark a function with multiple runs
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            runs: Number of benchmark runs
            warmup: Number of warmup runs (excluded from results)
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with benchmark statistics
        """
        func_name = func.__name__
        self.logger.info(f"Benchmarking {func_name} with {runs} runs (+{warmup} warmup)")
        
        times = []
        memory_deltas = []
        
        # Warmup runs
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Benchmark runs
        for run in range(runs):
            gc.collect()  # Clean up before each run
            
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            try:
                func(*args, **kwargs)
                success = True
            except Exception as e:
                self.logger.warning(f"Run {run+1} failed: {e}")
                success = False
            
            if success:
                end_time = time.time()
                end_memory = self.process.memory_info().rss / 1024 / 1024
                
                times.append(end_time - start_time)
                memory_deltas.append(end_memory - start_memory)
        
        if not times:
            return {'error': 'All benchmark runs failed'}
        
        # Calculate statistics
        times_array = np.array(times)
        memory_array = np.array(memory_deltas)
        
        return {
            'mean_time': np.mean(times_array),
            'median_time': np.median(times_array),
            'std_time': np.std(times_array),
            'min_time': np.min(times_array),
            'max_time': np.max(times_array),
            'mean_memory': np.mean(memory_array),
            'successful_runs': len(times),
            'total_runs': runs
        }
    
    def compare_functions(
        self, 
        func1: Callable, 
        func2: Callable, 
        *args, 
        runs: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare performance of two functions
        
        Args:
            func1: First function (e.g., original implementation)
            func2: Second function (e.g., optimized implementation)
            *args: Function arguments
            runs: Number of benchmark runs
            **kwargs: Function keyword arguments
            
        Returns:
            Comparison results with performance improvements
        """
        name1 = func1.__name__
        name2 = func2.__name__
        
        self.logger.info(f"Comparing {name1} vs {name2}")
        
        # Benchmark both functions
        results1 = self.benchmark_function(func1, *args, runs=runs, **kwargs)
        results2 = self.benchmark_function(func2, *args, runs=runs, **kwargs)
        
        if 'error' in results1 or 'error' in results2:
            return {'error': 'One or both functions failed during benchmarking'}
        
        # Calculate improvements
        time_improvement = (results1['mean_time'] - results2['mean_time']) / results1['mean_time'] * 100
        memory_improvement = (results1['mean_memory'] - results2['mean_memory']) / abs(results1['mean_memory']) * 100 if results1['mean_memory'] != 0 else 0
        
        return {
            'function1': {
                'name': name1,
                'mean_time': results1['mean_time'],
                'mean_memory': results1['mean_memory'],
                'std_time': results1['std_time']
            },
            'function2': {
                'name': name2,
                'mean_time': results2['mean_time'],
                'mean_memory': results2['mean_memory'],
                'std_time': results2['std_time']
            },
            'improvements': {
                'time_improvement_pct': time_improvement,
                'memory_improvement_pct': memory_improvement,
                'speedup_factor': results1['mean_time'] / results2['mean_time'],
                'time_saved_ms': (results1['mean_time'] - results2['mean_time']) * 1000
            },
            'verdict': {
                'faster': name2 if time_improvement > 0 else name1,
                'memory_efficient': name2 if memory_improvement > 0 else name1,
                'significant_improvement': time_improvement > 10  # 10% threshold
            }
        }
    
    def generate_report(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            include_details: Whether to include detailed metrics for each function
            
        Returns:
            Dictionary containing the performance report
        """
        if not self.metrics:
            return {'error': 'No metrics collected'}
        
        # Summary statistics
        all_times = [m['wall_time'] for m in self.metrics.values() if 'wall_time' in m]
        all_memory = [m['memory_delta'] for m in self.metrics.values() if 'memory_delta' in m]
        
        successful_functions = sum(1 for m in self.metrics.values() if m.get('success', True))
        total_functions = len(self.metrics)
        
        report = {
            'summary': {
                'total_functions_profiled': total_functions,
                'successful_executions': successful_functions,
                'success_rate': successful_functions / total_functions * 100,
                'total_execution_time': sum(all_times),
                'average_execution_time': np.mean(all_times) if all_times else 0,
                'total_memory_delta': sum(all_memory),
                'report_generated': datetime.now().isoformat()
            },
            'top_performers': {
                'fastest_function': min(self.metrics.items(), key=lambda x: x[1].get('wall_time', float('inf')))[0] if all_times else None,
                'slowest_function': max(self.metrics.items(), key=lambda x: x[1].get('wall_time', 0))[0] if all_times else None,
                'most_memory_efficient': min(self.metrics.items(), key=lambda x: x[1].get('memory_delta', float('inf')))[0] if all_memory else None,
                'most_memory_intensive': max(self.metrics.items(), key=lambda x: x[1].get('memory_delta', 0))[0] if all_memory else None
            }
        }
        
        if include_details:
            report['detailed_metrics'] = {}
            for func_name, metrics in self.metrics.items():
                report['detailed_metrics'][func_name] = {
                    'wall_time_ms': metrics.get('wall_time', 0) * 1000,
                    'cpu_time_ms': metrics.get('cpu_time', 0) * 1000,
                    'memory_delta_mb': metrics.get('memory_delta', 0),
                    'success': metrics.get('success', True),
                    'timestamp': metrics.get('timestamp', '').isoformat() if metrics.get('timestamp') else ''
                }
        
        return report
    
    def print_report(self, sort_by: str = 'wall_time', top_n: Optional[int] = None):
        """
        Print formatted performance report
        
        Args:
            sort_by: Metric to sort by ('wall_time', 'memory_delta', 'cpu_time')
            top_n: Number of top functions to display (None for all)
        """
        if not self.metrics:
            print("❌ No performance metrics collected")
            return
        
        print("\n" + "="*80)
        print("🚀 PERFORMANCE PROFILER REPORT")
        print("="*80)
        
        # Summary
        report = self.generate_report(include_details=False)
        summary = report['summary']
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   • Functions Profiled: {summary['total_functions_profiled']}")
        print(f"   • Success Rate: {summary['success_rate']:.1f}%")
        print(f"   • Total Execution Time: {summary['total_execution_time']:.3f}s")
        print(f"   • Average Execution Time: {summary['average_execution_time']:.3f}s")
        print(f"   • Total Memory Delta: {summary['total_memory_delta']:+.1f}MB")
        
        # Top performers
        top = report['top_performers']
        print(f"\n🏆 TOP PERFORMERS:")
        if top['fastest_function']:
            fastest_time = self.metrics[top['fastest_function']]['wall_time']
            print(f"   • Fastest: {top['fastest_function']} ({fastest_time:.3f}s)")
        if top['slowest_function']:
            slowest_time = self.metrics[top['slowest_function']]['wall_time']
            print(f"   • Slowest: {top['slowest_function']} ({slowest_time:.3f}s)")
        
        # Detailed metrics
        print(f"\n📋 DETAILED METRICS (sorted by {sort_by}):")
        print("-" * 80)
        
        # Sort metrics
        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        
        if top_n:
            sorted_metrics = sorted_metrics[:top_n]
        
        for func_name, metrics in sorted_metrics:
            status = "✅" if metrics.get('success', True) else "❌"
            wall_time = metrics.get('wall_time', 0)
            memory_delta = metrics.get('memory_delta', 0)
            cpu_time = metrics.get('cpu_time', 0)
            
            print(f"{status} {func_name:<30} | "
                  f"Time: {wall_time:7.3f}s | "
                  f"CPU: {cpu_time:7.3f}s | "
                  f"Memory: {memory_delta:+7.1f}MB")
        
        print("="*80)
    
    def save_report(self, filepath: str):
        """
        Save performance report to JSON file
        
        Args:
            filepath: Path to save the report
        """
        report = self.generate_report(include_details=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {filepath}")
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.metrics.clear()
        self.memory_snapshots.clear()
        self.logger.info("Performance metrics cleared")


class CachePerformanceMonitor:
    """
    Specialized monitor for caching system performance
    """
    
    def __init__(self):
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'total_time_saved': 0,
            'cache_operations': []
        }
    
    def record_cache_hit(self, time_saved: float):
        """Record a cache hit with time saved"""
        self.cache_stats['hits'] += 1
        self.cache_stats['total_requests'] += 1
        self.cache_stats['total_time_saved'] += time_saved
        
        self.cache_stats['cache_operations'].append({
            'type': 'hit',
            'time_saved': time_saved,
            'timestamp': datetime.now()
        })
    
    def record_cache_miss(self, load_time: float):
        """Record a cache miss with load time"""
        self.cache_stats['misses'] += 1
        self.cache_stats['total_requests'] += 1
        
        self.cache_stats['cache_operations'].append({
            'type': 'miss',
            'load_time': load_time,
            'timestamp': datetime.now()
        })
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.cache_stats['total_requests'] == 0:
            return 0.0
        return self.cache_stats['hits'] / self.cache_stats['total_requests']
    
    def get_time_savings(self) -> float:
        """Get total time saved by caching"""
        return self.cache_stats['total_time_saved']
    
    def print_cache_report(self):
        """Print cache performance report"""
        stats = self.cache_stats
        hit_rate = self.get_hit_rate()
        
        print("\n" + "="*60)
        print("💾 CACHE PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n📊 CACHE STATISTICS:")
        print(f"   • Total Requests: {stats['total_requests']}")
        print(f"   • Cache Hits: {stats['hits']}")
        print(f"   • Cache Misses: {stats['misses']}")
        print(f"   • Hit Rate: {hit_rate:.1%}")
        print(f"   • Total Time Saved: {stats['total_time_saved']:.2f}s")
        
        if stats['total_requests'] > 0:
            avg_time_saved = stats['total_time_saved'] / stats['hits'] if stats['hits'] > 0 else 0
            print(f"   • Average Time Saved per Hit: {avg_time_saved:.3f}s")
        
        print("="*60)


# Example usage and demonstration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Example: Profile a data loading function
    @profiler.profile_function("optimized_data_loading")
    def example_data_loading():
        """Example function to demonstrate profiling"""
        time.sleep(0.1)  # Simulate data loading
        data = pd.DataFrame(np.random.randn(1000, 10))
        return data.sum().sum()
    
    @profiler.profile_function("slow_data_loading")
    def example_slow_loading():
        """Example slow function for comparison"""
        time.sleep(0.3)  # Simulate slower loading
        data = pd.DataFrame(np.random.randn(1000, 10))
        return data.sum().sum()
    
    # Demonstrate profiling
    print("🚀 Running performance profiling demonstration...")
    
    # Profile functions
    result1 = example_data_loading()
    result2 = example_slow_loading()
    
    # Show individual reports
    profiler.print_report()
    
    # Demonstrate function comparison
    print("\n🔍 Comparing function performance...")
    comparison = profiler.compare_functions(
        example_slow_loading, 
        example_data_loading, 
        runs=3
    )
    
    if 'error' not in comparison:
        improvements = comparison['improvements']
        print(f"\n⚡ PERFORMANCE COMPARISON RESULTS:")
        print(f"   • Speed Improvement: {improvements['time_improvement_pct']:.1f}%")
        print(f"   • Speedup Factor: {improvements['speedup_factor']:.2f}x")
        print(f"   • Time Saved: {improvements['time_saved_ms']:.1f}ms per call")
        print(f"   • Winner: {comparison['verdict']['faster']}")
    
    print("\n✅ Performance profiling demonstration completed!")