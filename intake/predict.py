"""Competition entry: official readers -> model -> official result writer."""
import argparse
import csv
import os
import sys
from pathlib import Path
import torch
from .common import ROOT, official_module
from .features import extract_runs
from .model import load_checkpoint, predict_run, deduplicate


def predict(data_root, checkpoint, output, device='cpu', threshold=None, threads=4):
    torch.set_num_threads(threads)
    data_root,output=Path(data_root),Path(output)
    if not data_root.is_dir():raise FileNotFoundError(str(data_root))
    input_api,output_api=official_module('input'),official_module('output')
    # The reader preserves duplicate/conflicting metadata. Baseline does not condition on it.
    info=input_api.InfoReader(data_root/'userinfobean.txt')
    model,ckpt=load_checkpoint(checkpoint,device)
    cfg=ckpt['config'];infer=cfg['inference']
    threshold=infer['score_threshold'] if threshold is None else threshold
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.pending')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        csv.writer(f).writerow(output_api.HEADER)
    log=[]
    try:
        for person in sorted(p for p in data_root.iterdir() if p.is_dir()):
            person_info=info.read_info(person.name)
            intervals=[];files=0;runs_count=0
            for file_id,text in input_api.DataReader(person).iter_data():
                files+=1
                runs=extract_runs(text,cfg['preprocess'],is_text=True)
                del text
                for run in runs:
                    runs_count+=1;run['id']=file_id
                    features=run.pop('features')
                    intervals+=predict_run(model,features,run,ckpt['normalization'],threshold,
                                            infer['min_duration_ms'])
            intervals=deduplicate(intervals,infer['nms_iou'])
            for a,b,score in intervals:
                output_api.save_result(person.name,int(a),int(b),1,output_path=temporary)
            log.append({'subject':person.name,'files':files,'valid_runs':runs_count,
                        'metadata_records':len(person_info),'eating_intervals':len(intervals)})
            print(log[-1],flush=True)
        count=output_api.validate_result(temporary)
        os.replace(str(temporary),str(output))  # Replace prior run, including empty predictions.
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print('Saved %d eating intervals to %s' % (count,output),flush=True)
    return log


def main():
    # Bundled EXE looks beside itself, never relative to the caller's working directory.
    base=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else ROOT
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=base/'ACC_test_data')
    p.add_argument('--checkpoint',type=Path,default=base/'model.pt')
    p.add_argument('--output',type=Path,default=base/'results.csv')
    p.add_argument('--device',default='cpu');p.add_argument('--threshold',type=float)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();predict(a.data_root,a.checkpoint,a.output,a.device,a.threshold,a.threads)

if __name__=='__main__':main()
