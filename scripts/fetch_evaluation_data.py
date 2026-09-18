#!/usr/bin/env python3
"""Fetch the pinned WikiText-2 raw test split for reproducible quality diagnostics.
Run: uv run --frozen --extra hf --with pyarrow python scripts/fetch_evaluation_data.py
Data remains under ignored work/. See the source dataset card for attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = 'Salesforce/wikitext'
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
CHECKSUMS = {
    'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91',
    'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7',
    'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c',
}

def fetch(split):
    file = f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
    source = Path(hf_hub_download(REPO, file, repo_type='dataset', revision=REVISION))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CHECKSUMS[split]:
        raise ValueError('pinned dataset checksum mismatch')
    out = Path('work/evaluation-data'); out.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(pq.read_table(source, columns=['text'])['text'].to_pylist())
    target = out / f'wikitext-2-{split}.txt'; target.write_text(text)
    metadata = {'repository': REPO, 'revision': REVISION, 'file': file,
        'source_sha256': digest, 'text_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'license': 'CC-BY-SA-3.0 / GFDL; see https://huggingface.co/datasets/Salesforce/wikitext',
        'split': split, 'characters': len(text)}
    (out / f'wikitext-2-{split}-source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=sorted(CHECKSUMS), default=['test'])
    for split in dict.fromkeys(parser.parse_args().splits):
        fetch(split)

if __name__ == '__main__':
    main()
