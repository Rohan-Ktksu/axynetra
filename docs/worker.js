/* Real Python ML execution off the UI thread. No uploaded data leaves this worker. */
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
const DATA_URL = 'https://raw.githubusercontent.com/Rohan-Ktksu/axynetra/main/reference/data/';
let runtime;
let currentAction = 'train';
let trained = false;
let tested = false;
let busy = false;
self.report_progress = (fraction, text) => self.postMessage({kind:'progress', action:currentAction, fraction, text});

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function initialize() {
  if (runtime) return;
  self.report_progress(0.01, 'Downloading the Python WebAssembly runtime (first visit only)');
  const {loadPyodide} = await import(PYODIDE_URL + 'pyodide.mjs');
  runtime = await loadPyodide({indexURL:PYODIDE_URL, stdout:()=>{}, stderr:()=>{}});
  self.report_progress(0.02, 'Loading NumPy, pandas, scikit-learn and Matplotlib');
  await runtime.loadPackage(['numpy','pandas','scikit-learn','matplotlib']);
  runtime.FS.mkdirTree('/axynetra/reference/data');
  runtime.FS.chdir('/axynetra');
  for (const name of ['multiclass_intrusion.py','website_core.py','browser_runner.py']) {
    runtime.FS.writeFile('/axynetra/' + name, await fetchBytes(new URL('runtime/' + name, self.location.href)));
  }
  const names = ['DATA_PROVENANCE.json','UNSW_NB15_training_clean.csv','UNSW_NB15_testing_no_overlap.csv'];
  await Promise.all(names.map(async name => {
    self.report_progress(0.03, `Downloading bundled reference data: ${name}`);
    runtime.FS.writeFile('/axynetra/reference/data/' + name, await fetchBytes(DATA_URL + name));
  }));
  await runtime.runPythonAsync("import sys\nsys.path.insert(0, '/axynetra')\nfrom browser_runner import browser_train, browser_test, browser_predict, browser_template");
}

self.onmessage = async ({data}) => {
  if (busy) return self.postMessage({kind:'error',action:data.action,text:'An operation is already running.'});
  busy = true;
  currentAction = data.action;
  try {
    if (data.action === 'train') {
      await initialize();
      const result = JSON.parse(await runtime.runPythonAsync('browser_train()'));
      trained = true;
      self.postMessage({kind:'trained',result});
    } else if (data.action === 'test') {
      if (!trained || tested) throw new Error('Train first. Held-out evaluation runs once per experiment.');
      const result = JSON.parse(await runtime.runPythonAsync('browser_test()'));
      tested = true;
      self.postMessage({kind:'tested',result});
    } else if (data.action === 'predict') {
      if (!tested) throw new Error('Train and test the model first.');
      runtime.globals.set('request_json', JSON.stringify(data.input));
      const result = JSON.parse(await runtime.runPythonAsync('browser_predict(request_json)'));
      self.postMessage({kind:'predicted',result});
    } else if (data.action === 'template') {
      if (!trained) throw new Error('Train first to generate the matching template.');
      self.postMessage({kind:'template',csv:await runtime.runPythonAsync('browser_template()')});
    } else throw new Error('Unknown operation.');
  } catch (error) {
    if (!trained && data.action === 'train') runtime = undefined;
    self.postMessage({kind:'error',action:data.action,text:String(error)});
  } finally { busy = false; }
};
