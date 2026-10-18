import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from intake.common import ROOT, official_module, read_json, read_config
from intake.features import extract_runs, feature_names
from intake.model import make_model, model_item, predict_run, deduplicate, padded_length, load_checkpoint
from intake.evaluate import event_metrics, checkpoint_selection_key
from intake.predict import predict
from intake.prepare import prepare

BASE=1784472600000


def signal_text(offset=0, bins=24):
    out=io.StringIO();w=csv.writer(out,delimiter='\t')
    w.writerow(['ACC_TIME','PPG_TIME','GYRO_TIME','ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z'])
    for i in range(bins):
        t=BASE+offset+i*5000
        # Repeated timestamps contain multiple signal samples; zero PPG time is missing.
        w.writerow([t,0,t,1+i,2,3,4,5,6]);w.writerow([t,0,t,3+i,2,3,4,5,6])
    return out.getvalue()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg=read_json(ROOT/'configs/intake_actionformer.json')
        self.cfg['preprocess']['ppg_channels']=0

    def test_timestamp_bins_gaps_empty_and_modalities(self):
        runs=extract_runs(signal_text(),self.cfg['preprocess'],True)
        self.assertEqual(len(runs),1);self.assertEqual(runs[0]['features'].shape,(27,24))
        self.assertEqual(runs[0]['grid_start_ms'],BASE)
        self.assertAlmostEqual(runs[0]['features'][0,0],(np.log(2)+np.log(4))/2,places=5)
        self.assertTrue((runs[0]['features'][-1]==0).all())
        second=signal_text(offset=1000000).split('\n',1)[1]
        self.assertEqual(len(extract_runs(signal_text()+second,self.cfg['preprocess'],True)),2)
        self.assertEqual(extract_runs('',self.cfg['preprocess'],True),[])
        full=extract_runs(signal_text(),read_json(ROOT/'configs/intake_actionformer.json')['preprocess'],True)
        self.assertEqual(full[0]['features'].shape[0],67)
        self.assertTrue((full[0]['features'][24:64]==0).all())

    def test_original_actionformer_positive_negative_backward_and_absolute_time(self):
        run=extract_runs(signal_text(bins=64),self.cfg['preprocess'],True)[0]
        x=run.pop('features');run['segments_ms']=[[BASE+20000,BASE+90000]]
        norm={'mean':[0]*27,'std':[1]*27}
        item=model_item(x,run,norm,True)
        np.testing.assert_allclose((item['segments'].numpy()+.5)*5000+BASE,run['segments_ms'])
        model=make_model(self.cfg,27,128);model.train()
        negative=dict(item,segments=torch.empty((0,2)),labels=torch.empty((0,),dtype=torch.long))
        loss=model([item,negative]);self.assertGreater(float(loss['reg_loss']),0)
        self.assertTrue(torch.isfinite(loss['final_loss']))
        loss['final_loss'].backward()
        self.assertIsNotNone(model.cls_head.cls_head.conv.weight.grad)
        model.eval()
        predictions=predict_run(model,x,run,norm,0,0)
        for a,b,s in predictions:self.assertTrue(BASE<=a<b<=run['valid_end_ms'])

    def test_nms_and_event_matching(self):
        self.assertEqual(len(deduplicate([(0,10,.9),(1,9,.8),(20,30,.7)])),2)
        metrics=event_metrics({'A':[(0,10,.9),(1,9,.8)]},{'A':[(0,10)]})
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(1,1,0))

    def test_official_interfaces_empty_results_and_multiple_files(self):
        api=official_module('input');out=official_module('output')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);person=root/'HNU00001';person.mkdir()
            (person/'b.txt').write_text(signal_text(),encoding='utf-8')
            (person/'a.txt').write_text('',encoding='utf-8')
            self.assertEqual([k for k,v in api.DataReader(person).iter_data()],['a','b'])
            results=root/'results.csv'
            with results.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerow(out.HEADER)
            self.assertEqual(out.validate_result(results),0)
            out.save_result('HNU00001',BASE,BASE+1000,1,results)
            self.assertEqual(out.validate_result(results),1)
            with self.assertRaises(ValueError):out.save_result('HNU00001',BASE//1000,BASE//1000+1,1,results)

    def test_competition_entry_traversal_and_fresh_output(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['inference']['min_duration_ms']=0
        model=make_model(cfg,27,128)
        with torch.no_grad():
            model.cls_head.cls_head.conv.bias.fill_(10)
            model.reg_head.offset_head.conv.bias.fill_(3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'ACC_test_data';data.mkdir()
            (data/'userinfobean.txt').write_text('externalid,height,weight,age,gender\nHNU00001,170,60,23,M\n',encoding='utf-8')
            person=data/'HNU00001';person.mkdir();(data/'HNU00002').mkdir()
            (person/'one.txt').write_text(signal_text(),encoding='utf-8')
            (person/'two.txt').write_text(signal_text(offset=1000000),encoding='utf-8')
            checkpoint=root/'model.pt'
            torch.save({'config':cfg,'input_dim':27,'max_seq_len':128,'model':model.state_dict(),
                        'normalization':{'mean':[0]*27,'std':[1]*27}},checkpoint)
            output=root/'results.csv'
            log=predict(data,checkpoint,output,threshold=.9,threads=2)
            self.assertEqual(log[0]['files'],2);self.assertEqual(log[1]['files'],0)
            self.assertGreater(official_module('output').validate_result(output),0)
            predict(data,checkpoint,output,threshold=1.0,threads=2)
            self.assertEqual(official_module('output').validate_result(output),0)

    def test_prepare_excludes_exact_ids_and_groups_aliases_without_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'Data';data.mkdir()
            cfg=root/'config.json';cfg.write_text(__import__('json').dumps(self.cfg))
            (data/'excluded_subjects.json').write_text('{"excluded_externalids":["HNU21026j"]}')
            sensors=[];meals=[]
            for i,(e,h) in enumerate([('HNU21026','accountA'),('HNU00004','accountA'),
                                        ('HNU00002','accountB'),('HNU00003','accountC')]):
                folder=data/'IMU+PPG'/('sensorData-%d'%i);folder.mkdir(parents=True)
                (folder/'sample.txt').write_text(signal_text(),encoding='utf-8')
                sensors.append({'externalid':e,'healthid':h,'sensorData':'sensorData-%d.zip'%i,
                                'timeStamp.startTime':'2026-07-19 22:50:00',
                                'timeStamp.endTime':'2026-07-19 23:00:00'})
                meals.append({'externalid':e,'healthid':h,'timeFrame.startTime':BASE+20000,
                              'timeFrame.endTime':BASE+80000,'beforeTime':BASE+20000,'afterTime':BASE+80000})
            # Same healthid as a retained participant: no healthid expansion of exclusion.
            sensors.append(dict(sensors[0],externalid='HNU21026j',sensorData='missing_excluded.zip'))
            for name,rows in [('IMU+PPG.csv',sensors),('MealInfo.csv',meals)]:
                with (data/name).open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            manifest=prepare(data,root/'cache',cfg)
            self.assertEqual(len(manifest['records']),4)
            self.assertNotIn('HNU21026J',{r['subject'] for r in manifest['records']})
            self.assertIn('HNU21026',{r['subject'] for r in manifest['records']})
            self.assertEqual(len({r['split'] for r in manifest['records'] if r['healthid']=='accountA'}),1)
            for a in ['train','validation','test']:
                for b in ['train','validation','test']:
                    if a!=b:
                        self.assertFalse({r['group'] for r in manifest['records'] if r['split']==a} &
                                         {r['group'] for r in manifest['records'] if r['split']==b})

    def test_event_iou_threshold_is_strictly_greater_than_quarter(self):
        for end, expected_tp in [(4001,0),(4000,0),(3999,1)]:
            result=event_metrics({'A':[(BASE,BASE+end,.9)]},{'A':[(BASE,BASE+1000)]})
            self.assertEqual(result['tp'],expected_tp)
            self.assertEqual(result['fp'],1-expected_tp)
            self.assertEqual(result['fn'],1-expected_tp)
            self.assertEqual(result['iou_comparison'],'>')
            self.assertEqual(result['iou_threshold'],.25)

    def test_event_sensitivity_ppv_f1_and_matched_boundary_mae(self):
        predictions={'A':[(BASE+1000,BASE+12000,.9),(BASE+30000,BASE+40000,.8)],
                     'B':[(BASE+17000,BASE+34000,.9)]}
        targets={'A':[(BASE,BASE+10000)],
                 'B':[(BASE+20000,BASE+30000),(BASE+50000,BASE+60000)]}
        result=event_metrics(predictions,targets)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(2,1,1))
        for key in ['sensitivity','positive_predictive_value','f1']:
            self.assertAlmostEqual(result[key],2/3)
        self.assertAlmostEqual(result['start_mae_seconds'],2.)
        self.assertAlmostEqual(result['end_mae_seconds'],3.)
        self.assertAlmostEqual(result['boundary_mae_seconds'],2.5)
        for pred,gt in [({},{}),({'A':[(BASE,BASE+1000,.9)]},{}),({},targets)]:
            empty=event_metrics(pred,gt)
            self.assertEqual(empty['f1'],0)
            self.assertIsNone(empty['start_mae_seconds'])
            self.assertIsNone(empty['end_mae_seconds'])
            self.assertIsNone(empty['boundary_mae_seconds'])

    def test_checkpoint_selection_prioritizes_f1_then_mae(self):
        self.assertGreater(checkpoint_selection_key({'f1':.9,'boundary_mae_seconds':100}),
                           checkpoint_selection_key({'f1':.8,'boundary_mae_seconds':1}))
        self.assertGreater(checkpoint_selection_key({'f1':.9,'boundary_mae_seconds':1}),
                           checkpoint_selection_key({'f1':.9,'boundary_mae_seconds':2}))
        self.assertEqual(checkpoint_selection_key({'f1':0.,'boundary_mae_seconds':None}),
                         (0.,float('-inf')))

    def test_tridet_dispatch_backward_background_and_checkpoint(self):
        cfg=read_config(ROOT/'experiment261016/intake_tridet.yaml')
        cfg['preprocess']['ppg_channels']=0
        run=extract_runs(signal_text(bins=64),cfg['preprocess'],True)[0]
        x=run.pop('features');run['segments_ms']=[[BASE+20000,BASE+90000]]
        norm={'mean':[0]*27,'std':[1]*27}
        item=model_item(x,run,norm,True)
        model=make_model(cfg,27,padded_length(128,cfg))
        self.assertEqual(type(model).__name__,'TriDet')
        self.assertTrue(model.use_trident_head)
        self.assertEqual(type(model.backbone).__name__,'SGPBackbone')
        negative=dict(item,segments=torch.empty((0,2)),labels=torch.empty((0,),dtype=torch.long))
        for items in [[item,negative],[negative]]:
            model.train();model.zero_grad()
            losses=model(items)
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            losses['final_loss'].backward()
            self.assertIsNotNone(model.cls_head.cls_head.conv.weight.grad)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'tridet.pt'
            torch.save({'config':cfg,'input_dim':27,'max_seq_len':128,'model':model.state_dict(),
                        'normalization':norm},checkpoint)
            restored,_=load_checkpoint(checkpoint)
            self.assertEqual(type(restored).__name__,'TriDet')
            predictions=predict_run(restored,x,run,norm,0,0)
            for a,b,score in predictions:
                self.assertTrue(BASE<=a<b<=run['valid_end_ms'])
        # Same interpreter can still instantiate ActionFormer with its own registries.
        self.assertEqual(type(make_model(self.cfg,27,128)).__name__,'PtTransformer')

if __name__=='__main__':unittest.main()
