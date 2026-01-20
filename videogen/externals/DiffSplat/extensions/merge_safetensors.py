import os
import argparse
import safetensors
import safetensors.torch


def main():
    parser = argparse.ArgumentParser(description="Merge multiple safetensors in a directory into a single safetensors")
    parser.add_argument("root", type=str, help="Root directory containing safetensors")

    args = parser.parse_args()
    safetensor_file_paths = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".safetensors")]
    if len(safetensor_file_paths) == 1:
        return

    tensors = {}
    for path in safetensor_file_paths:
        with safetensors.safe_open(path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    safetensors.torch.save_file(tensors, os.path.join(args.root, os.path.basename(safetensor_file_paths[0]).split("-")[0]) + ".safetensors")

    for f in os.listdir(args.root):
        path = os.path.join(args.root, f)
        if path.endswith(".index.json"):
            os.remove(path)
        if path in safetensor_file_paths:
            os.remove(path)


if __name__ == "__main__":
    main()
