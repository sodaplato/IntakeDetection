import csv
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(encoding))))
        except UnicodeDecodeError:
            continue
    raise ValueError('Cannot decode CSV: %s' % path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_config(path):
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping: %s' % path)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def official_module(name):
    # Load the supplied interfaces unchanged, including when bundled by PyInstaller.
    path = ROOT / '00 TaskRequirments' / (name + '.py')
    spec = importlib.util.spec_from_file_location('competition_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject_id(value):
    return ''.join(value.split()).upper()
