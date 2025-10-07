import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def analyze_performance_metrics(csv_path):
    df = pd.read_csv(csv_path)
    
    print("=== PERFORMANCE ANALYSIS ===\n")
    
    # Training speed analysis
    print(" TRAINING SPEED:")
    total_time = df['epoch_time_s'].sum()
    avg_epoch_time = df['epoch_time_s'].mean()
    fastest_epoch = df['epoch_time_s'].min()
    slowest_epoch = df['epoch_time_s'].max()
    
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average epoch time: {avg_epoch_time:.1f}s")
    print(f"  Fastest epoch: {fastest_epoch:.1f}s")
    print(f"  Slowest epoch: {slowest_epoch:.1f}s")
    print(f"  Time per epoch range: {slowest_epoch-fastest_epoch:.1f}s variance")
    
    # Memory usage analysis
    print(f"\n MEMORY USAGE:")
    avg_memory = df['peak_mem_mb'].mean()
    max_memory = df['peak_mem_mb'].max()
    min_memory = df['peak_mem_mb'].min()
    
    print(f"  Average peak memory: {avg_memory:.1f} MB ({avg_memory/1024:.2f} GB)")
    print(f"  Maximum peak memory: {max_memory:.1f} MB ({max_memory/1024:.2f} GB)")
    print(f"  Minimum peak memory: {min_memory:.1f} MB ({min_memory/1024:.2f} GB)")
    print(f"  Memory stability: {max_memory-min_memory:.1f} MB variance")
    
    # Model size analysis
    params = df['trainable_params'].iloc[0]
    print(f"\n MODEL SIZE:")
    print(f"  Trainable parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Memory per parameter: {max_memory*1e6/params:.1f} bytes/param")
    print(f"  Parameters per MB: {params/(max_memory):.0f} params/MB")
    
    # Efficiency metrics
    print(f"\n EFFICIENCY METRICS:")
    params_per_sec = params / avg_epoch_time
    mem_efficiency = params / max_memory
    print(f"  Parameter updates per second: {params_per_sec:.0f}")
    print(f"  Memory efficiency: {mem_efficiency:.0f} params/MB")
    
    # Plot performance metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training time per epoch
    ax1.plot(df['epoch'], df['epoch_time_s'], 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Training Time per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=avg_epoch_time, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_epoch_time:.1f}s')
    ax1.legend()
    
    # Peak memory usage
    ax2.plot(df['epoch'], df['peak_mem_mb'], 'g-o', linewidth=2, markersize=4)
    ax2.set_title('Peak GPU Memory Usage')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=avg_memory, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_memory:.1f}MB')
    ax2.legend()
    
    # Training efficiency (accuracy gain per time)
    time_cumsum = df['epoch_time_s'].cumsum()
    ax3.plot(time_cumsum, df['train_acc'], 'purple', linewidth=2, label='Train Acc')
    ax3.plot(time_cumsum, df['val_acc'], 'orange', linewidth=2, label='Val Acc')
    ax3.set_title('Accuracy vs Training Time')
    ax3.set_xlabel('Cumulative Time (seconds)')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Memory vs Accuracy
    scatter = ax4.scatter(df['peak_mem_mb'], df['val_acc'], c=df['epoch'], cmap='viridis', s=50)
    ax4.set_title('Memory Usage vs Validation Accuracy')
    ax4.set_xlabel('Peak Memory (MB)')
    ax4.set_ylabel('Validation Accuracy')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    plt.show()

def benchmark_inference_speed(model, head, test_loader, device, num_samples=100):
    """Benchmark inference speed."""
    model.eval()
    head.eval()
    
    print("\n=== INFERENCE SPEED BENCHMARK ===")
    
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i * images.size(0) >= num_samples:
                break
                
            images = images.to(device)
            
            # Warm up GPU on the first batch
            if i == 0:
                for _ in range(5):
                    _ = get_logits_fn(model, head, images)
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = get_logits_fn(model, head, images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
    
    avg_batch_time = np.mean(times)
    std_batch_time = np.std(times)
    images_per_sec = test_loader.batch_size / avg_batch_time
    
    print(f"  Average batch time: {avg_batch_time*1000:.2f} ± {std_batch_time*1000:.2f} ms")
    print(f"  Images per second: {images_per_sec:.1f}")
    print(f"  Time per image: {avg_batch_time*1000/test_loader.batch_size:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="Analyze training and inference performance from a metrics file.")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to the metrics.csv file to analyze.")
    # Add other arguments if needed, e.g., for the inference benchmark
    # parser.add_argument("--run-inference", action="store_true", help="Run inference speed benchmark.")
    
    args = parser.parse_args()

    print(f"Running performance analysis on: {args.csv_path}")
    
    if not Path(args.csv_path).exists():
        print(f"Error: File not found at {args.csv_path}")
        return

    analyze_performance_metrics(args.csv_path)

   
if __name__ == "__main__":
    main()
