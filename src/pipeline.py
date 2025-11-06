import argparse
from . import extract, validate, feature_engineering

def parse_args():
    p = argparse.ArgumentParser(description="Stage A pipeline runner")
    p.add_argument("--step", choices=["extract", "validate", "features", "all"], default="all")
    return p.parse_args()

def run_step(step: str) -> None:
    if step == "extract":
        extract.run()
    elif step == "validate":
        validate.run()
    elif step == "features":
        feature_engineering.run()
    elif step == "all":
        extract.run()
        validate.run()
        feature_engineering.run()

def main():
    args = parse_args()
    run_step(args.step)

if __name__ == "__main__":
    main()