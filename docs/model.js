fetch('site-data.json').then(response=>{if(!response.ok)throw Error('Unable to load feature schema.');return response.json();}).then(data=>{
  const target=document.getElementById('feature-list'); target.replaceChildren();
  const table=document.createElement('table');const head=table.createTHead().insertRow();
  ['Feature','Type','Group','Processing'].forEach(name=>{const th=document.createElement('th');th.scope='col';th.textContent=name;head.append(th);});
  const body=table.createTBody();data.features.forEach(feature=>{const row=body.insertRow();for(const key of ['feature','type','group','processing'])row.insertCell().textContent=feature[key];});target.append(table);
  data.groups.forEach(group=>{const details=document.createElement('details'),summary=document.createElement('summary'),p=document.createElement('p'),pre=document.createElement('pre');summary.textContent=group.group;p.textContent=group.meaning;pre.textContent=group.features.join(', ');pre.className='plain-code';details.append(summary,p,pre);document.getElementById('feature-groups').append(details);});
  Object.entries(data.models).forEach(([name,settings])=>{const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=name+' · exact settings';pre.className='plain-code';pre.textContent=settings+'\nBrowser override: n_jobs=1 where supported.';details.append(summary,pre);document.getElementById('model-settings').append(details);});
}).catch(error=>{document.getElementById('feature-list').textContent=error.message;});
