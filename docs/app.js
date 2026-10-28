const $ = id => document.getElementById(id);
let worker;
let predictions = [];
let busy = false;
let hasTrained = false;
let hasTested = false;
const urls = [];

function table(target, rows) {
  target.replaceChildren();
  if (!rows.length) { target.textContent = 'No records.'; return; }
  const columns = Object.keys(rows[0]);
  const grid = document.createElement('table');
  const head = grid.createTHead().insertRow();
  columns.forEach(column => { const th = document.createElement('th'); th.textContent = column; th.scope = 'col'; head.append(th); });
  const body = grid.createTBody();
  rows.forEach(row => { const tr = body.insertRow(); columns.forEach(column => {
    const cell = tr.insertCell(); const value = row[column];
    cell.textContent = value == null ? '—' : typeof value === 'number' && !Number.isInteger(value) ? Number(value.toPrecision(5)).toString() : String(value);
  }); });
  target.append(grid);
}

function metrics(target, values) {
  target.replaceChildren();
  for (const [label, value] of values) {
    const card = document.createElement('div');
    const strong = document.createElement('strong'); strong.textContent = value;
    const text = document.createElement('span'); text.textContent = label;
    card.append(strong,text); target.append(card);
  }
}

function blobUrl(content, type) {
  const url = URL.createObjectURL(new Blob([content],{type})); urls.push(url); return url;
}
function download(content, name, type) {
  const a = document.createElement('a'); a.href = blobUrl(content,type); a.download = name; a.click();
}
function csv(rows) {
  const keys = Object.keys(rows[0] || {});
  const cell = value => {
    let text = value == null ? '' : String(value);
    if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"','""') + '"';
  };
  return [keys.map(cell).join(','),...rows.map(row => keys.map(key=>cell(row[key])).join(','))].join('\r\n');
}
function error(text) { $('error').hidden = false; $('error').textContent = text; }
function buttons() {
  $('train').disabled = busy || hasTrained;
  $('test').disabled = busy || !hasTrained || hasTested;
  $('prediction-controls').disabled = busy || !hasTested;
}
function start(action, input) {
  if (busy) return;
  busy = true; buttons(); $('error').hidden = true;
  if (action === 'train' || action === 'test') $(action + '-progress').hidden = false;
  if (action === 'predict') { $('prediction-output').hidden = true; $('predict-status').textContent = 'Predicting through the fitted pipeline…'; }
  worker.postMessage({action,input});
}

async function initializePage() {
  const response = await fetch('site-data.json');
  if (!response.ok) throw new Error('Dataset preview could not be loaded. Refresh to retry.');
  const data = await response.json();
  table($('data-preview'),data.preview);
  data.classes.forEach(name=>{ const option = new Option(name,name); $('category').add(option); });
  if (!window.Worker || !window.WebAssembly) throw new Error('This experiment requires a current browser with WebAssembly and Web Worker support.');
  worker = new Worker('worker.js',{type:'module'});
  worker.onerror = event => { busy=false; buttons(); error('The model worker stopped: ' + event.message + '. Reload this page to start a new experiment.'); };
  worker.onmessage = ({data}) => {
    if (data.kind === 'progress') {
      $(data.action + '-status').textContent = data.text;
      $(data.action + '-bar').value = data.fraction;
      if (data.action === 'train') { const li = document.createElement('li'); li.textContent = data.text; $('train-log').append(li); li.scrollIntoView({block:'nearest'}); }
      return;
    }
    busy = false;
    if (data.kind === 'error') {
      error(data.text);
      if (data.action === 'predict') $('predict-status').textContent = 'Prediction failed. Check the message below.';
      buttons(); return;
    }
    if (data.kind === 'trained') {
      hasTrained = true;
      $('train-badge').textContent = 'TRAINING COMPLETE';
      $('train-step').classList.add('current');
      $('training-output').hidden = false;
      table($('leaderboard'),data.result.leaderboard);
      $('winner').textContent = `${data.result.best_model} selected by validation Macro F1. Ready for held-out testing.`;
      $('runtime-note').textContent = `This run: Python ${data.result.python} · scikit-learn ${data.result.sklearn} · seed 42 · one browser thread. Model and uploaded data stay in this tab. Reloading starts a new experiment.`;
      $('testing').classList.remove('locked'); $('test-badge').textContent = 'READY TO TEST';
    }
    if (data.kind === 'tested') {
      hasTested = true;
      const result = data.result, meta = result.meta, m = meta.heldout_test_evaluation;
      try {
        sessionStorage.setItem('axynetra-evaluation', JSON.stringify({
          meta: {...meta, evaluated_at: new Date().toISOString()},
          images: Object.fromEntries(Object.entries(result.images).map(([name, value])=>[name, 'data:image/png;base64,' + value])),
          report: result.report, auc: result.auc,
        }));
      } catch {
        $('runtime-note').textContent += ' This browser could not retain the results for the Model & Features page; that page will show the published reference run.';
      }
      $('test-badge').textContent = 'TESTING COMPLETE'; $('test-step').classList.add('current');
      $('test-output').hidden = false;
      metrics($('test-metrics'),[['Held-out accuracy',(m.accuracy*100).toFixed(1)+'%'],['Macro F1',m.macro_f1.toFixed(4)],['Weighted F1',m.weighted_f1.toFixed(4)],['Test records',meta.test_rows.toLocaleString()]]);
      for (const [id,key] of [['confusion','confusion_matrix'],['roc','roc_curves'],['importance','feature_importance']]) {
        $(id).hidden = !result.images[key]; if (result.images[key]) $(id).src = 'data:image/png;base64,' + result.images[key];
      }
      $('report').textContent = result.report;
      $('importance-source').textContent = 'Source model: ' + meta.feature_importance_model + '. If the winner is not a tree, the best compatible validation tree supplies this output.';
      table($('auc'),result.auc); table($('alerts'),result.alerts);
      $('download-artifacts').href = blobUrl(Uint8Array.from(atob(result.zip),c=>c.charCodeAt(0)),'application/zip');
      $('prediction').classList.remove('locked'); $('predict-badge').textContent = 'READY TO PREDICT';
    }
    if (data.kind === 'predicted') {
      predictions = data.result.results;
      $('predict-status').textContent = `Prediction complete · ${predictions.length} records`;
      $('prediction-output').hidden = false; $('predict-step').classList.add('current');
      const correct = predictions.filter(row=>row.correct!=null);
      metrics($('prediction-metrics'),[
        ['Predicted attacks',predictions.filter(r=>r.predicted_attack!=='Normal').length],
        ['Predicted Normal',predictions.filter(r=>r.predicted_attack==='Normal').length],
        ['Flagged for review',predictions.filter(r=>r.needs_human_review).length],
        ['Sample agreement',correct.length?(100*correct.filter(r=>r.correct).length/correct.length).toFixed(1)+'%':'No labels supplied']]);
      table($('predictions'),predictions); table($('scores'),data.result.scores); table($('prediction-inputs'),data.result.inputs);
    }
    if (data.kind === 'template') download(data.csv,'axynetra_input_template.csv','text/csv');
    buttons();
  };
  buttons();
}

$('train').onclick=()=>start('train');
$('test').onclick=()=>start('test');
$('predict').onclick=async()=>{
  if ($('prediction-source').value === 'upload') {
    const file = $('upload').files[0];
    if (!file) return error('Choose a CSV file first.');
    if (file.size > 5*1024*1024) return error('Upload a CSV of at most 5 MB.');
    start('predict',{csv:await file.text()});
  } else {
    const count = Number($('count').value);
    if (!Number.isInteger(count) || count<1 || count>100) return error('Choose between 1 and 100 records.');
    start('predict',{category:$('category').value,count});
  }
};
$('template').onclick=()=>start('template');
$('download-predictions').onclick=()=>download(csv(predictions),'axynetra_predictions.csv','text/csv');
for (const id of ['prediction-source','category','count','upload']) $(id).onchange=()=>{
  $('prediction-output').hidden=true; $('predict-status').textContent='';
  const upload = $('prediction-source').value==='upload';
  $('upload-box').hidden=!upload; $('category-label').hidden=upload; $('count-label').hidden=upload;
};
window.addEventListener('beforeunload',()=>urls.forEach(url=>URL.revokeObjectURL(url)));
initializePage().catch(e=>error(e.message));
