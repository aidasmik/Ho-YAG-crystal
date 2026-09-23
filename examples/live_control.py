"""Send a local pause/resume/step/cancel command to a bounded Stage 7W worker."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hoyag.live_control import request_control


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_directory',type=Path)
    p.add_argument('command',choices=('pause','resume','step','cancel'))
    args=p.parse_args()
    print(request_control(args.case_directory,args.command))


if __name__=='__main__':main()
