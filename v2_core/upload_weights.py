"""
RunPod Weights Uploader
=======================
Since `runpodctl` is not installed on the local Windows machine, and Git LFS 
crashes VSCode on massive binary files, we use pure python to SFTP the weights 
directly to the active RunPod instance.
"""

import sys
import paramiko
from pathlib import Path
from loguru import logger

def upload_weights(
    instance_ip: str, 
    instance_port: int, 
    instance_user: str = "root"
):
    """
    Connects to the RunPod SSH endpoint and uploads the 1.5GB .pt file.
    
    Args:
        instance_ip: The TCP IP provided by the RunPod Connect menu (e.g. 12.34.56.78)
        instance_port: The TCP Port provided by the RunPod Connect menu (e.g. 12345)
    """
    logger.info("Initializing Secure P2P SFTP Transfer...")
    
    # Locate the massive weights file
    local_path = Path(__file__).parent / "frc_pretrained_core_A6000_FINAL.pt"
    if not local_path.exists():
        logger.error(f"Cannot find weights at {local_path}")
        sys.exit(1)
        
    remote_path = "/workspace/SNAP-C1/v2_core/frc_pretrained_core_A6000_FINAL.pt"
    
    # Locate the user's default SSH Key
    key_path = Path.home() / ".ssh" / "id_ed25519"
    if not key_path.exists():
        # Fallback to RSA if ed25519 doesn't exist
        key_path = Path.home() / ".ssh" / "id_rsa"
        
    if not key_path.exists():
        logger.critical("No SSH keys found in ~/.ssh/ directory! Please configure RunPod SSH keys or use the Web drag-and-drop.")
        sys.exit(1)

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        logger.info(f"Connecting to {instance_user}@{instance_ip}:{instance_port} using {key_path.name}...")
        ssh.connect(
            hostname=instance_ip,
            port=instance_port,
            username=instance_user,
            key_filename=str(key_path)
        )
        
        logger.success("Secure connection established!")
        
        sftp = ssh.open_sftp()
        
        # Determine file size for progress printing
        file_size = local_path.stat().st_size
        
        def progress_callback(transferred: int, total: int):
            # Print a clean progress bar
            percent = (transferred / total) * 100
            sys.stdout.write(f"\rUploading SOTA Weights: [{percent:.1f}%] of {(file_size / (1024**3)):.2f} GB")
            sys.stdout.flush()
            
        logger.info(f"Commencing binary streaming to {remote_path}...")
        sftp.put(str(local_path), remote_path, callback=progress_callback)
        print("\n")
        
        sftp.close()
        ssh.close()
        
        logger.success("Transfer Complete! You can now start the RLFS script on the RunPod.")
        
    except Exception as e:
        logger.error(f"Transfer failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Securely upload SNAP-C1 weights to RunPod.")
    parser.add_argument("--ip", type=str, required=True, help="The RunPod SSH IP address")
    parser.add_argument("--port", type=int, required=True, help="The RunPod SSH Port")
    
    args = parser.parse_args()
    
    upload_weights(instance_ip=args.ip, instance_port=args.port)
