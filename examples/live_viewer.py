"""Monitor accepted Stage 7W snapshots; display frames never advance the solver."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hoyag.replay_viewer import monitor_live


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot_directory',type=Path)
    p.add_argument('--target-fps',type=float,default=30.)
    p.add_argument('--seconds',type=float)
    p.add_argument('--screenshot',type=Path)
    p.add_argument('--view',choices=('overview','thermal','optics'),default='overview')
    args=p.parse_args()
    result=monitor_live(args.snapshot_directory,target_fps=args.target_fps,
        seconds=args.seconds,off_screen=args.screenshot is not None,
        screenshot=args.screenshot,view=args.view)
    print(json.dumps(result))


if __name__=='__main__':main()
