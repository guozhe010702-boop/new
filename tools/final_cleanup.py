from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
subprocess.run(['systemctl','stop','advanced-algorithms.service'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run([sys.executable,str(ROOT/'tools/control.py'),'restore'],check=True)
