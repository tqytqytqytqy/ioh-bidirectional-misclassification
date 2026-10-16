from pathlib import Path
import hashlib
import json

R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((R/'qc/full_qc_aggregate.json').read_text())
assert q['all_expected_samples_decoded'] and q['source_changes']==0
assert q['code_sha256']==sha(R/'qc/run_full_qc.py')
assert q['core_code_sha256']==sha(R/'qc/full_qc_core.py')
assert q['qc_input_sha256']==sha(R/'private/qc_inputs.npz')
files={n:sha(R/'private'/n) for n in ['qc_inputs.npz','records.json']}
manifest={'status':'ready','input_sha256':files,'contract_sha256':sha(R/'analysis_contract.json'),
 'source_full_decode_complete':True,'clinical_adjudication':False,'scientific_source_changes':False}
(R/'private/qc_ready.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'ready','inputs':list(files),'contract_frozen':True}))
