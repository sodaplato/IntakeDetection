from pathlib import Path
import zipfile,json,time,re,shutil
ROOT=Path(__file__).resolve().parents[1]
log=[];seen=set()
def decoded(s):
 if '#U' in s:return re.sub(r'#U([0-9a-fA-F]{4})',lambda m:chr(int(m[1],16)),s)
 try:return s.encode('cp437').decode('gbk')
 except (UnicodeError,LookupError):return s
while True:
 archives=sorted((p for p in ROOT.rglob('*.zip') if p not in seen),key=lambda p:(len(p.parts),str(p)))
 if not archives:break
 for p in archives:
  seen.add(p)
  existing=list(ROOT.glob(p.stem+'_20*')) if p.parent==ROOT else []
  base=existing[0] if existing else p.with_suffix('')
  written=skipped=0
  try:
   with zipfile.ZipFile(p) as z:
    infos=z.infolist()
    for i in infos:
     parts=Path(i.filename).parts
     if Path(i.filename).is_absolute() or '..' in parts:raise ValueError('Unsafe archive path')
     target=base
     for part in parts:
      choices=[part,decoded(part),''.join('#U%04x'%ord(c) if ord(c)>127 else c for c in decoded(part))]
      found=next((target/c for c in choices if (target/c).exists()),None)
      target=found if found else target/decoded(part)
     if i.is_dir():target.mkdir(parents=True,exist_ok=True);continue
     if target.exists():
      if target.stat().st_size==i.file_size:skipped+=1;continue
      raise ValueError('Existing file differs: '+str(target))
     target.parent.mkdir(parents=True,exist_ok=True)
     with z.open(i) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
     written+=1
   log.append(dict(archive=str(p.relative_to(ROOT)),destination=str(base.relative_to(ROOT)),written=written,skipped=skipped,status='ok'))
  except Exception as e:
   log.append(dict(archive=str(p.relative_to(ROOT)),status='error',error=str(e)))
  if len(log)%50==0:print('Processed',len(log),'archives; latest',p.name,flush=True)
  (ROOT/'reports/extraction_manifest.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
print('DONE',len(log),'archives',sum(x.get('written',0) for x in log),'new files','errors',sum(x['status']=='error' for x in log),flush=True)
