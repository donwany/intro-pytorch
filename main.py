import torch 


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Is MPS available?", torch.backends.mps.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("CUDA available?", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
