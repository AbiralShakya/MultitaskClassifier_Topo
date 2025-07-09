# Checkpoint Training Guide

## 🚀 **How to Use Checkpoint Training**

The checkpoint training system automatically saves your training progress every few epochs, so you can recover from crashes and continue training without losing work.

### **Quick Start**

```bash
# Start checkpoint training
python multitask_ti_classification/training/checkpoint_training.py
```

### **What It Does**

1. **Automatic Checkpointing**: Saves progress every 5 epochs by default
2. **Crash Recovery**: Can resume from the last checkpoint if training crashes
3. **Memory Management**: Better GPU memory management to prevent segmentation faults
4. **Progress Tracking**: Saves training/validation losses and model state

### **Checkpoint Files Created**

When you run checkpoint training, it creates:

```
./checkpoints/
├── checkpoint_epoch_5.pt      # Full training state (epoch 5)
├── checkpoint_epoch_10.pt     # Full training state (epoch 10)
├── indices_epoch_5.pkl        # Dataset split indices
├── indices_epoch_10.pkl       # Dataset split indices
└── ...
```

### **Resuming Training**

When you restart the script, it will:

1. **Detect existing checkpoints** automatically
2. **Ask if you want to resume**: `Do you want to resume from this checkpoint? (y/n):`
3. **Load everything**: Model state, optimizer state, training progress, dataset splits
4. **Continue from where you left off**

### **Example Usage**

#### **First Run (Fresh Start)**
```bash
python multitask_ti_classification/training/checkpoint_training.py
```
Output:
```
Starting checkpoint-based training loop...
Using device: cuda:0
Starting fresh training...
Loading dataset with preloading...
Data loaders created. Train: 2030 batches, Val: 435 batches
Starting epoch 1/100
...
```

#### **After a Crash (Resume)**
```bash
python multitask_ti_classification/training/checkpoint_training.py
```
Output:
```
Starting checkpoint-based training loop...
Using device: cuda:0
Found existing checkpoint: ./checkpoints/checkpoint_epoch_15.pt
Do you want to resume from this checkpoint? (y/n): y
Checkpoint loaded from epoch 15 with best val loss: 0.2345
Resuming from epoch 16
Starting epoch 16/100
...
```

### **Customization Options**

You can modify the checkpoint behavior in the script:

```python
# Change checkpoint frequency (save every N epochs)
checkpoint_training_loop(checkpoint_frequency=10)  # Save every 10 epochs

# Change checkpoint directory
checkpoint_training_loop(checkpoint_dir="./my_checkpoints")
```

### **What Gets Saved**

Each checkpoint contains:
- **Model state**: All model weights and parameters
- **Optimizer state**: Learning rate, momentum, etc.
- **Scheduler state**: Learning rate scheduling information
- **Training progress**: Current epoch, best validation loss
- **Loss history**: Training and validation losses for all epochs
- **Dataset splits**: Train/val/test indices to ensure consistency
- **Configuration**: All config parameters

### **Memory Management Features**

The checkpoint training includes enhanced memory management:

1. **GPU Memory Monitoring**: Shows memory usage during training
2. **Automatic Cache Clearing**: Clears spectral encoder cache every 25 batches
3. **Memory Cleanup**: Clears GPU cache between epochs
4. **Garbage Collection**: Forces Python garbage collection

### **Error Recovery**

If training crashes with a segmentation fault:

1. **Don't panic!** Your progress is saved
2. **Restart the script**: It will detect the latest checkpoint
3. **Choose 'y' to resume**: Training continues from where it crashed
4. **Monitor memory**: The script includes better memory management

### **Best Practices**

1. **Use checkpoint training for long runs**: Especially with your 4-5 hour epochs
2. **Monitor GPU memory**: The script shows memory usage
3. **Let it save checkpoints**: Don't interrupt during checkpoint saving
4. **Keep checkpoints**: Don't delete checkpoint files unless you're sure

### **Troubleshooting**

#### **"No checkpoint found"**
- This is normal for the first run
- Training will start fresh

#### **"Checkpoint corrupted"**
- Delete the corrupted checkpoint file
- Resume from the previous checkpoint

#### **"Out of memory"**
- The script includes memory management
- If still happening, reduce batch size in config

### **Integration with GPU Spectral Encoder**

The checkpoint training works perfectly with the new GPU-accelerated spectral encoder:

- **Faster training**: GPU acceleration reduces epoch time
- **Better memory**: Automatic cache clearing prevents memory buildup
- **Crash recovery**: Checkpoints save progress even if segmentation faults occur

### **Expected Workflow**

1. **Start training**: `python multitask_ti_classification/training/checkpoint_training.py`
2. **Let it run**: Training saves checkpoints every 5 epochs
3. **If it crashes**: Restart the same command
4. **Resume**: Choose 'y' when prompted
5. **Continue**: Training picks up exactly where it left off

This system ensures you never lose training progress, even with the segmentation faults you've been experiencing! 