fetch('site-data.json').then(response=>{if(!response.ok)throw Error('Unable to load feature schema.');return response.json();}).then(data=>{
  const target=document.getElementById('feature-list'); target.replaceChildren();
  const table=document.createElement('table');const head=table.createTHead().insertRow();
  ['Feature','Type','Group','Processing'].forEach(name=>{const th=document.createElement('th');th.scope='col';th.textContent=name;head.append(th);});
  const body=table.createTBody();data.features.forEach(feature=>{const row=body.insertRow();for(const key of ['feature','type','group','processing'])row.insertCell().textContent=feature[key];});target.append(table);
  data.groups.forEach(group=>{const details=document.createElement('details'),summary=document.createElement('summary'),p=document.createElement('p'),pre=document.createElement('pre');summary.textContent=group.group;p.textContent=group.meaning;pre.textContent=group.features.join(', ');pre.className='plain-code';details.append(summary,p,pre);document.getElementById('feature-groups').append(details);});
  Object.entries(data.models).forEach(([name,settings])=>{const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=name+' · exact settings';pre.className='plain-code';pre.textContent=settings+'\nBrowser override: n_jobs=1 where supported.';details.append(summary,pre);document.getElementById('model-settings').append(details);});
}).catch(error=>{document.getElementById('feature-list').textContent=error.message;});

async function showEvaluation() {
  let result;
  try { result = JSON.parse(sessionStorage.getItem('axynetra-evaluation')); } catch { /* Use the published run when storage is unavailable. */ }
  const latest = Boolean(result?.meta?.heldout_test_evaluation && result?.images?.roc_curves);
  if (!latest) {
    const response = await fetch('evaluation/summary.json');
    if (!response.ok) throw Error('Evaluation results could not be loaded. Refresh to retry, or run Train Model and Test Model in the workspace.');
    result = await response.json();
  }
  const meta = result.meta, metric = meta.heldout_test_evaluation;
  const $ = id => document.getElementById(id);
  $('evaluation-source').textContent = latest ? 'YOUR LATEST BROWSER RUN' : 'PUBLISHED REFERENCE RUN';
  const when = meta.evaluated_at ? ' Evaluated ' + new Date(meta.evaluated_at).toLocaleString() + '.' : '';
  $('evaluation-note').textContent = `${latest ? 'Your latest completed test in this browser tab' : 'Saved evaluation on the verified, cleaned reference CSVs'} · selected model: ${meta.best_model}.${when} These results describe the cleaned, sampled, modified UNSW-NB15 split.`;
  const metrics = [
    ['Held-out accuracy', (metric.accuracy * 100).toFixed(1) + '%'],
    ['Macro F1', metric.macro_f1.toFixed(4)],
    ['Weighted F1', metric.weighted_f1.toFixed(4)],
    ['Macro precision', metric.macro_precision.toFixed(4)],
    ['Macro recall', metric.macro_recall.toFixed(4)],
    ['Macro ROC-AUC', metric.roc_auc_ovr_macro == null ? 'Undefined' : metric.roc_auc_ovr_macro.toFixed(4)],
  ];
  $('evaluation-metrics').replaceChildren();
  for (const [label, value] of metrics) {
    const card = document.createElement('div'), strong = document.createElement('strong'), text = document.createElement('span');
    strong.textContent = value; text.textContent = label; card.append(strong, text); $('evaluation-metrics').append(card);
  }
  $('evaluation-protocol').textContent = `${meta.train_rows.toLocaleString()} training records (${meta.fit_train_rows.toLocaleString()} fit / ${meta.val_rows.toLocaleString()} validation) · ${meta.test_rows.toLocaleString()} held-out test records · ${meta.predictor_count} raw inputs. The winner was selected by validation Macro F1; test results did not select it.`;
  for (const [id, name] of [['evaluation-roc', 'roc_curves'], ['evaluation-confusion', 'confusion_matrix'], ['evaluation-importance', 'feature_importance']]) {
    $(id).hidden = !result.images[name];
    if (result.images[name]) $(id).src = result.images[name];
  }
  $('evaluation-report').textContent = result.report;
  $('evaluation-importance-note').textContent = `Source model: ${meta.feature_importance_model}. Global impurity reduction is associative, not causal, and does not explain individual predictions.`;
  $('evaluation-auc-note').textContent = `AUC defined for ${metric.roc_auc_defined_classes}/${metric.roc_auc_total_classes} classes. Full macro requires all ten classes; model scores are uncalibrated.`;
  const grid = document.createElement('table'), head = grid.createTHead().insertRow();
  for (const label of ['Class', 'Test records', 'ROC-AUC', 'Status']) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; head.append(th); }
  const body = grid.createTBody();
  for (const row of result.auc) {
    const tr = body.insertRow();
    for (const value of [row.class, row.support, row.roc_auc_ovr == null ? 'Undefined' : row.roc_auc_ovr.toFixed(4), row.status]) tr.insertCell().textContent = value;
  }
  $('evaluation-auc').replaceChildren(grid);
  $('evaluation-content').hidden = false;
}
showEvaluation().catch(error=>{document.getElementById('evaluation-note').textContent=error.message;document.getElementById('evaluation-source').textContent='RESULTS UNAVAILABLE';});
