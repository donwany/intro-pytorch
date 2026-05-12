import torch 

if __name__ == "__main__":
        # Example tensor operation to test GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.rand(10000, 10000, device=device)
        y = torch.rand(10000, 10000, device=device)
        z = x + y
        print("Tensor operation successful on GPU.")
        print(f"results {z}")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
        x = torch.rand(10000, 10000, device=device)
        y = torch.rand(10000, 10000, device=device)
        z = x + y
        print("Tensor operation successful on CPU.")
        print(f"results {z}")