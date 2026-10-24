"""Repack existing FP32 frozen features without running the visual encoder."""
import argparse
import json
from pathlib import Path
from vjepa_policy.history_features import sha256_file
from vjepa_policy.packed_history_features import pack_frozen_cache, warm_packed_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warm-only", action="store_true",
                        help="Warm existing packed observations before torchrun; do not repack")
    args = parser.parse_args()
    if args.warm_only:
        packed = Path(args.output) / "manifest.json"
        if json.loads(packed.read_text())["source_manifest_sha256"] != sha256_file(args.manifest):
            raise ValueError("Packed source manifest bytes changed")
        print(json.dumps(warm_packed_observations(packed)), flush=True)
    else:
        pack_frozen_cache(args.manifest, args.output)


if __name__ == "__main__":
    main()
