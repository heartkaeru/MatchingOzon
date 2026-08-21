"""
Inference entry point script.
CLI: -i/--input, -m/--model, -o/--output
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for MatchingOzon")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON/Parquet file")
    parser.add_argument("-o", "--output", required=True, help="Path to save predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Running inference. Input: {args.input}, Output: {args.output}")
    # TODO: Load weights, run model, write submission format

if __name__ == "__main__":
    main()
