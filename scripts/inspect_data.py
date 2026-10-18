from pathlib import Path
import csv,json,collections,datetime,statistics
R=Path(__file__).resolve().parents[1];tz=datetime.timezone(datetime.timedelta(hours=8))
def dt(x):return datetime.datetime.fromtimestamp(int(x)/1000,tz).isoformat()
meta={}
for p in R.glob('*_20*/*.csv'):
 rows=list(csv.DictReader(p.open(encoding='gb18030')));meta['sensor' if p.name=='IMU+PPG.csv' else 'meal' if '9910' in p.name else 'user']=(p,rows)
s={}
for k,(p,rows) in meta.items():
 s[k]={'path':str(p.relative_to(R)),'rows':len(rows),'participants':len({r['externalid'] for r in rows if r['externalid']}),'columns':list(rows[0]),'missing':{c:sum(not r[c] for r in rows) for c in rows[0]},'categories':{c:dict(collections.Counter(r[c] for r in rows)) for c in ['dietaryType','tablewareType','dietaryHand','wearHand','deviceType','sensorType'] if c in rows[0]}}
meals=meta['meal'][1];sensor=meta['sensor'][1]
starts=[int(r['timeFrame.startTime']) for r in meals];ends=[int(r['timeFrame.endTime']) for r in meals];duration=[(b-a)/60000 for a,b in zip(starts,ends)]
s['meal'].update(start=dt(min(starts)),end=dt(max(ends)),duration_minutes={'min':min(duration),'median':statistics.median(duration),'max':max(duration),'invalid':sum(d<=0 for d in duration)})
for key,col in [('sensor','sensorData'),('meal','diningPictures')]:
 names={p.name for p in R.rglob('*.zip')};s[key]['missing_archives']=[r[col].split('/')[-1] for r in meta[key][1] if r[col].split('/')[-1] not in names]
s['sensor']['start']=min(r['timeStamp.startTime'] for r in sensor);s['sensor']['end']=max(r['timeStamp.endTime'] for r in sensor)
s['participant_sets']={k:sorted({r['externalid'] for r in rows if r['externalid']}) for k,(p,rows) in meta.items()}
s['meal_without_sensor_participant']=sorted(set(s['participant_sets']['meal'])-set(s['participant_sets']['sensor']))
# Coverage using metadata times; union intervals avoids double-counting overlapping recordings.
intervals=collections.defaultdict(list)
for r in sensor:
 try:
  a=datetime.datetime.strptime(r['timeStamp.startTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()*1000;b=datetime.datetime.strptime(r['timeStamp.endTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()*1000
  if b>a:intervals[r['externalid']].append((a,b))
 except ValueError:pass
coverage=[]
for r,a,b in zip(meals,starts,ends):
 spans=sorted((max(a,x),min(b,y)) for x,y in intervals[r['externalid']] if x<b and y>a);merged=[]
 for x,y in spans:
  if merged and x<=merged[-1][1]:merged[-1][1]=max(y,merged[-1][1])
  else:merged.append([x,y])
 covered=sum(y-x for x,y in merged);coverage.append(covered/(b-a) if b>a else 0)
s['meal']['metadata_coverage']={'full':sum(x>=.99999 for x in coverage),'partial':sum(0<x<.99999 for x in coverage),'none':sum(x==0 for x in coverage)}
(R/'reports/data_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a not in ['columns','participant_sets','missing_archives','path']} for k,v in s.items() if isinstance(v,dict) and k!='participant_sets'},ensure_ascii=False,indent=2))
