from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
targets={ROOT/'build'/n for n in ('advanced_vision','model_batch','sr_tiled','hardware_tests')}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if (p/'exe').resolve() in targets:raise SystemExit('Refusing to rebuild active executable PID '+p.name)
    except (PermissionError,FileNotFoundError,ProcessLookupError):pass
