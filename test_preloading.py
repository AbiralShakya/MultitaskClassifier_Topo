#!/usr/bin/env python3
"""
Test script to verify dataset preloading functionality and measure performance improvement.
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multitask_ti_classification.helper import config
from multitask_ti_classification.helper.dataset import MaterialDataset

def test_preloading_performance():
    """Test the performance difference between preloaded and non-preloaded datasets."""
    
    print("Testing dataset preloading performance...")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Master index path: {config.MASTER_INDEX_PATH}")
    print(f"K-space graphs dir: {config.KSPACE_GRAPHS_DIR}")
    print(f"DOS/Fermi dir: {config.DOS_FERMI_DIR}")
    
    # Test without preloading
    print("\n=== Testing WITHOUT preloading ===")
    start_time = time.time()
    dataset_no_preload = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=False
    )
    init_time_no_preload = time.time() - start_time
    print(f"Initialization time (no preload): {init_time_no_preload:.2f} seconds")
    
    # Test loading first few samples without preloading
    start_time = time.time()
    for i in range(min(10, len(dataset_no_preload))):
        sample = dataset_no_preload[i]
        if i % 5 == 0:
            print(f"  Loaded sample {i}: {sample['jid']}")
    load_time_no_preload = time.time() - start_time
    print(f"Time to load 10 samples (no preload): {load_time_no_preload:.2f} seconds")
    
    # Test with preloading
    print("\n=== Testing WITH preloading ===")
    start_time = time.time()
    dataset_with_preload = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=True
    )
    init_time_with_preload = time.time() - start_time
    print(f"Initialization time (with preload): {init_time_with_preload:.2f} seconds")
    
    # Test loading first few samples with preloading
    start_time = time.time()
    for i in range(min(10, len(dataset_with_preload))):
        sample = dataset_with_preload[i]
        if i % 5 == 0:
            print(f"  Loaded sample {i}: {sample['jid']}")
    load_time_with_preload = time.time() - start_time
    print(f"Time to load 10 samples (with preload): {load_time_with_preload:.2f} seconds")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    print(f"Dataset size: {len(dataset_with_preload)} samples")
    print(f"Preloading overhead: {init_time_with_preload - init_time_no_preload:.2f} seconds")
    print(f"Sample loading speedup: {load_time_no_preload / load_time_with_preload:.1f}x")
    
    if init_time_with_preload < init_time_no_preload + load_time_no_preload:
        print("✅ Preloading is beneficial for this dataset size!")
    else:
        print("⚠️  Preloading overhead might be too high for small datasets")
    
    # Memory usage estimation
    if hasattr(dataset_with_preload, 'cached_data') and dataset_with_preload.cached_data:
        sample_size = len(dataset_with_preload.cached_data[0])
        print(f"Estimated memory per sample: ~{sample_size} tensors")
        print(f"Total cached samples: {len(dataset_with_preload.cached_data)}")
    
    return dataset_with_preload

if __name__ == "__main__":
    try:
        dataset = test_preloading_performance()
        print("\n✅ Preloading test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during preloading test: {e}")
        import traceback
        traceback.print_exc() 