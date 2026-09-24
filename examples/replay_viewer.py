"""Replay one hash-verified completed Stage 7W case using optional PyVista."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hoyag.snapshots import snapshot_from_case
from hoyag.replay_viewer import render_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_directory',type=Path)
    p.add_argument('--screenshot',type=Path)
    p.add_argument('--mode',type=int,default=0)
    p.add_argument('--polarization',type=int,choices=(0,1),default=0)
    p.add_argument('--deformation-exaggeration',type=float,default=1.)
    p.add_argument('--view',choices=('overview','thermal','optics'),default='overview')
    args=p.parse_args()
    snapshot=snapshot_from_case(args.case_directory)
    print(render_replay(snapshot,off_screen=args.screenshot is not None,
        screenshot=args.screenshot,mode=args.mode,polarization=args.polarization,
        deformation_exaggeration=args.deformation_exaggeration,view=args.view))


if __name__=='__main__':main()
