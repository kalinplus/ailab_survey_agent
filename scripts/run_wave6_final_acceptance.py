import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/Users/kalin/github/AILabMacroHard')
PYTHON = '/Users/kalin/miniconda3/bin/python'
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
prefix = os.environ.get('EVISURVEY_ACCEPT_PREFIX', 'wave6-final')
archive = ROOT / 'archive' / (f'{prefix}-{stamp}')
marker_name = f"{prefix.replace('-', '_')}_archive_path.txt"
archive.mkdir(parents=True)
env = os.environ.copy()
env.update({
    'EVISURVEY_WRITER_LLM': '1', 'EVISURVEY_REPAIR_AGENT': '1',
    'EVISURVEY_REAL_NLI': '1', 'EVISURVEY_NLI_DEVICE': 'cpu',
    'HF_HUB_OFFLINE': '1', 'TOOL_TIMEOUT_SECONDS': '7200',
    'FINAL_SEED_PAPERS': '0', 'DYLD_FALLBACK_LIBRARY_PATH': '/opt/homebrew/lib',
    'HEAVY_LLM_MODEL': 'deepseek-flash', 'HEAVY_LLM_THINKING_EFFORT': 'disabled',
})
command = [PYTHON, 'main.py', '--topic', 'World Models for Games: A Survey',
           '--language', 'en', '--max-papers', '60', '--max-core-papers', '15',
           '--mode', 'full', '--use-mineru']
config = {'command': command, 'environment': {k: env[k] for k in [
    'EVISURVEY_WRITER_LLM','EVISURVEY_REPAIR_AGENT','EVISURVEY_REAL_NLI',
    'EVISURVEY_NLI_DEVICE','HF_HUB_OFFLINE','TOOL_TIMEOUT_SECONDS',
    'FINAL_SEED_PAPERS','DYLD_FALLBACK_LIBRARY_PATH','HEAVY_LLM_MODEL',
    'HEAVY_LLM_THINKING_EFFORT']}, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'started_at': datetime.datetime.now().astimezone().isoformat()}
(archive / 'run_config.json').write_text(json.dumps(config, indent=2))
(ROOT / f'logs/{marker_name}').write_text(str(archive))
print('Archive:', archive, flush=True)
with (archive/'run.log').open('w') as log:
    pipeline = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
print('Pipeline exit:', pipeline.returncode, flush=True)
# Freeze actual inputs and outputs before the evaluator creates anything else.
for folder in ['cache', 'output', 'requests', 'memory']:
    source=ROOT/folder
    if source.exists():
        shutil.copytree(source, archive/folder)
required=['output/survey.md','cache/evidence_store.json','cache/paper_cards.json',
          'cache/taxonomy.json','cache/figure_bank.json','cache/table_bank.json']
missing=[f for f in required if not (archive/f).exists()]
if missing:
    raise RuntimeError('Missing final artifacts: '+repr(missing))
# Require a freshly written survey; a failed early run must not evaluate an old file.
started=datetime.datetime.fromisoformat(config['started_at']).timestamp()
if (ROOT/'output/survey.md').stat().st_mtime < started:
    raise RuntimeError('Pipeline did not produce a new survey; old artifacts are archived only')
files={str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
       for folder in ['cache','output','requests','memory']
       for p in (archive/folder).rglob('*') if p.is_file()}
(archive/'manifest.json').write_text(json.dumps(files,indent=2))
eval_command=[PYTHON, 'scripts/run_survey_eval.py',
    '--survey',str(archive/'output/survey.md'),
    '--evidence-store',str(archive/'cache/evidence_store.json'),
    '--papers',str(archive/'cache/paper_cards.json'),
    '--taxonomy',str(archive/'cache/taxonomy.json'),
    '--figure-bank',str(archive/'cache/figure_bank.json'),
    '--table-bank',str(archive/'cache/table_bank.json'),
    '--gold-refs',str(archive/'cache/gold_refs.json'),
    '--seed-bibs',str(archive/'cache/seed_survey_bibs.json'),
    '--canonical',str(archive/'cache/canonical_papers.json'),
    '--ab-report',str(archive/'no-current-ab-report.json'),
    '--uncited-cache',str(archive/'uncited_claim_cache.json'),
    '--out',str(archive/'survey_eval_report')]
with (archive/'eval.log').open('w') as log:
    evaluation=subprocess.run(eval_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
result={'pipeline_exit':pipeline.returncode,'evaluation_exit':evaluation.returncode,
        'archive':str(archive),'completed_at':datetime.datetime.now().astimezone().isoformat()}
(archive/'execution_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
sys.exit(evaluation.returncode)
