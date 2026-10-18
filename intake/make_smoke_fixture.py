"""Create an organizer-layout fixture from cached validation source files (no labels)."""
import argparse
import os
import shutil
from pathlib import Path
from .common import ROOT, read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=read_json(a.cache/'manifest.json')
    if a.output.exists():raise ValueError('Choose a fresh fixture output')
    a.output.mkdir(parents=True)
    shutil.copyfile(str(ROOT/'Data/UserInfo.csv'),str(a.output/'userinfobean.txt'))
    seen=set()
    for r in m['records']:
        if r['split']!='validation' or r['source'] in seen:continue
        seen.add(r['source']);dest=a.output/r['subject']/Path(r['source']).name
        dest.parent.mkdir(exist_ok=True)
        try:os.link(r['source'],dest)
        except OSError:shutil.copyfile(r['source'],dest)
    print('Created fixture with %d complete TXT files (no labels)' % len(seen))

if __name__=='__main__':main()
