"""Run the same verified noise analysis used by the GUI."""
import argparse
from pathlib import Path
from rawrig.noise_analysis import analyze_noise_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    result = analyze_noise_session(args.session, progress=lambda message, *args: print(message, flush=True))
    print(result)
