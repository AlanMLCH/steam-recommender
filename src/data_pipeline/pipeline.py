import argparse
from . import extract, validate
from ..features import main as features_main

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
        features_main.run()
    elif step == "all":
        extract.run()
        validate.run()
        features_main.run()

def main():
    args = parse_args()
    run_step(args.step)

if __name__ == "__main__":
    main()