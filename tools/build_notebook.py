"""Build the notebook from its percent-format Python source; optionally execute it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / 'Axynetra_ML_Workflow.py').read_text(encoding='utf-8')
    cells = []
    kind, lines = None, []

    def flush():
        if kind is None:
            return
        content = '\n'.join(lines).strip('\n') + '\n'
        cell = {'cell_type': kind, 'metadata': {}, 'source': content.splitlines(keepends=True),
                'id': hashlib.sha256((str(len(cells)) + content).encode()).hexdigest()[:12]}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in source.splitlines():
        if line.startswith('# %%'):
            flush()
            kind = 'markdown' if '[markdown]' in line else 'code'
            lines = []
        elif kind == 'markdown':
            lines.append(line[2:] if line.startswith('# ') else line[1:] if line == '#' else line)
        elif kind == 'code':
            lines.append(line)
    flush()
    return {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3 (select Axynetra .venv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    notebook = build()
    path = ROOT / 'Axynetra_ML_Workflow.ipynb'
    if args.execute:
        import nbformat
        from nbclient import NotebookClient
        notebook = nbformat.reads(json.dumps(notebook), as_version=4)
        nbformat.validate(notebook)
        runtime = ROOT / '.temp' / 'jupyter'
        kernel = runtime / 'kernels' / 'axynetra'
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Axynetra project environment', 'language': 'python',
        }), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = str(runtime) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'runtime')
        os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
        NotebookClient(notebook, timeout=180, startup_timeout=120,
                       kernel_name='axynetra', resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.validate(notebook)
        assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code')
        assert not any(output.output_type == 'error' for cell in notebook.cells
                       if cell.cell_type == 'code' for output in cell.outputs)
        nbformat.write(notebook, path)
    else:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{path.name}: {len(notebook["cells"])} cells; executed={args.execute}')


if __name__ == '__main__':
    main()
