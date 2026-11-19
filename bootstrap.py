"""First-run installer and launcher. Run using Python 3.11 or 3.12."""
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parent

def main():
    if sys.version_info[:2] not in ((3,11),(3,12)):
        raise SystemExit('Please install Python 3.12 (or 3.11), then run this launcher again.')
    env=ROOT/'.venv'
    exe=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    marker=env/'.asl-ready'
    if not exe.exists():
        print('Creating local Python environment...',flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    if not marker.exists():
        print('Installing dependencies. Internet required for first setup...',flush=True)
        subprocess.run([str(exe),'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
        marker.write_text('ASL Studio dependencies installed\n')
    subprocess.run([str(exe),str(ROOT/'app.py'),'gui'],cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except (subprocess.CalledProcessError,OSError) as exc:
        raise SystemExit(f'Setup/launch failed: {exc}. See README.md for manual setup.')
