import argparse
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["miniwob", "mind2web"])
    parser.add_argument("--mind2web_data_dir", type=str)
    args = parser.parse_args()

    current_path = os.getcwd()
    memory_path = os.path.join(current_path, "memory")
    if args.env == "miniwob":
        from synapse.memory.miniwob.builde_memory import build_memory

        build_memory(memory_path)
    else:
        from synapse.memory.mind2web.build_memory import build_memory

        log_dir = Path(memory_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        build_memory(memory_path, args.mind2web_data_dir)
