"""Refresh relative artifact hashes and provenance notes for a completed run."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run-ids',nargs='+',required=True);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for run_id in a.run_ids:
 run=ROOT/'results'/'runs'/run_id;m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
 m['source_sha256']=sha(ROOT/'research_experiment.py')
 probe=m.setdefault('architecture_input_dependence_probe',{})
 probe.pop('before_training',None)
 probe['pretraining_probe_architectures']=['4:2:cnot_chain','4:2:cnot_star','6:2:cnot_chain']
 probe['post_hoc_probe_architectures']=['2:2:cnot_chain','2:2:cz_ring']
 probe['post_hoc_probes_used_for_model_selection']=False
 probe['criterion']='Each feature must have mean absolute readout change >1e-4 for a 0.2-radian perturbation under four random-weight draws and 32 random inputs per draw.'
 m['legacy_checkpoint']={'path':'trained_weights.npy','used_in_this_run':False,
   'note':'Legacy prototype file in the project root; excluded from the run package and not used as evidence because its provenance is unestablished.'}
 lim=m.setdefault('limitations',[])
 msg='Post-hoc ZNE-scale and calibration-budget sensitivity analyses reuse the original outer test folds and are exploratory, not independent validation.'
 if msg not in lim:lim.append(msg)
 artifacts={}
 for f in sorted(run.rglob('*')):
  if not f.is_file() or f.name in {'run_manifest.json','independent_audit.json'} or f.suffix.lower()=='.zip':continue
  artifacts[f.relative_to(ROOT).as_posix()]=sha(f)
 m['artifact_sha256']=artifacts
 (run/'run_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
 print(run_id,'source hash refreshed; hashed artifacts:',len(artifacts))
