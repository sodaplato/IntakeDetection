from pathlib import Path
import csv,json,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parents[1]
files=sorted(R.rglob('collect*.txt'))
if files:
 p=files[0]
 with p.open() as f:
  reader=csv.DictReader(f,delimiter='\t');rows=[]
  for i,r in enumerate(reader):
   rows.append(r)
   if i>=9999:break
 cols=list(rows[0]);(R/'reports/sensor_sample.csv').write_text(','.join(cols)+'\n'+'\n'.join(','.join(r[c] for c in cols) for r in rows[:10])+'\n')
 stats={'file':str(p.relative_to(R)),'sample_rows':len(rows),'columns':cols,'time_analysis':{}}
 for c in ['ACC_TIME','PPG_TIME','GYRO_TIME']:
  vals=[int(r[c]) for r in rows];diff=collections.Counter(b-a for a,b in zip(vals,vals[1:]));stats['time_analysis'][c]={'first':vals[0],'last':vals[-1],'most_common_deltas_ms':diff.most_common(8),'unique':len(set(vals))}
 stats['all_zero_in_sample']=[c for c in cols if all(float(r[c])==0 for r in rows)]
 (R/'reports/sensor_sample_analysis.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
 fig,axs=plt.subplots(3,1,figsize=(12,8))
 for ax,cc in zip(axs,[['ACC_X','ACC_Y','ACC_Z'],['GYRO_X','GYRO_Y','GYRO_Z'],['PPG1','PPG2','PPG3']]):
  for c in cc:ax.plot([float(r[c]) for r in rows[:2000]],label=c,linewidth=.7)
  ax.legend(loc='upper right');ax.grid(alpha=.2);ax.set_ylabel('Raw value')
 axs[-1].set_xlabel('Row index (timestamps repeat; sampling rate unverified)');fig.tight_layout();fig.savefig(str(R/'reports/sensor_preview.png'),dpi=140);plt.close(fig)
 print(json.dumps(stats,ensure_ascii=False,indent=2))
photos=sorted(p for p in R.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'] and 'reports' not in p.parts)
if photos:
 selected=photos[:8];canvas=Image.new('RGB',(1200,660),'white');draw=ImageDraw.Draw(canvas)
 for i,p in enumerate(selected):
  im=ImageOps.exif_transpose(Image.open(str(p)));im.thumbnail((290,285));x=(i%4)*300;y=(i//4)*330;canvas.paste(im,(x+(300-im.width)//2,y));draw.text((x+5,y+290),p.name[:35],fill='black')
 canvas.save(str(R/'reports/meal_photos_preview.jpg'))
 print('preview photos',len(selected))
