"""R11 Training Wrapper - unbuffered, with error handling."""
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Redirect to log file
log_path = os.path.join(os.path.dirname(__file__), 'r11_training.log')
log_file = open(log_path, 'w', buffering=1)  # line-buffered
sys.stdout = log_file
sys.stderr = log_file

try:
    # Set up path
    nexus_dir = os.path.join(os.path.dirname(__file__), 'SNAP-C1', 'nexus-r')
    sys.path.insert(0, nexus_dir)
    os.chdir(nexus_dir)
    
    from nexus_v1.training.train_bpe import train
    train()
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    log_file.close()
