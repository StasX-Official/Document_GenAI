const form = document.getElementById('generate-form');
const promptInput = document.getElementById('prompt');
const outputArea = document.getElementById('output');
const exportLinks = document.getElementById('export-links');
const downloadBtn = document.getElementById('download-btn');
const historyList = document.getElementById('history-list');
const refreshHistoryBtn = document.getElementById('refresh-history');
const enableEditingToggle = document.getElementById('enable-editing');

async function postForm(url, data) {
  const response = await fetch(url, {
    method: 'POST',
    body: data,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Generation failed');
  }
  return response.json();
}

function collectFormats() {
  return Array.from(form.querySelectorAll('input[name="formats"]:checked')).map((input) => input.value);
}

async function handleSubmit(event) {
  event.preventDefault();
  const formats = collectFormats();
  const data = new FormData();
  data.append('prompt', promptInput.value);
  data.append('model', form.model.value);
  data.append('formats', formats.join(','));
  data.append('enable_editing', enableEditingToggle.checked ? 'true' : 'false');

  form.classList.add('loading');
  try {
    const payload = await postForm('/api/generate', data);
    outputArea.value = payload.content;
    renderExportLinks(payload.files);
    await loadHistory();
  } catch (error) {
    alert(error.message);
  } finally {
    form.classList.remove('loading');
  }
}

function renderExportLinks(files) {
  exportLinks.innerHTML = '';
  Object.entries(files).forEach(([format, path]) => {
    const link = document.createElement('a');
    link.href = `file://${path}`;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = `${format.toUpperCase()} · Відкрити`;
    exportLinks.appendChild(link);
  });
}

function downloadEditedContent() {
  const blob = new Blob([outputArea.value], { type: 'text/plain;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `edited_document_${Date.now()}.txt`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function loadHistory(limit = 10) {
  const response = await fetch(`/api/history?limit=${limit}`);
  if (!response.ok) return;
  const payload = await response.json();
  historyList.innerHTML = '';
  payload.history.forEach((item) => {
    const li = document.createElement('li');
    const title = document.createElement('div');
    title.textContent = `${item.timestamp} · ${item.model}`;
    const formats = document.createElement('div');
    formats.textContent = `Формати: ${item.formats.join(', ')}`;
    const link = document.createElement('a');
    link.href = `file://${item.filepath}`;
    link.textContent = 'Відкрити документ';
    link.target = '_blank';
    li.append(title, formats, link);
    historyList.appendChild(li);
  });
}

function registerShortcuts() {
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      form.requestSubmit();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      downloadEditedContent();
    }
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'h') {
      event.preventDefault();
      loadHistory();
    }
  });
}

form.addEventListener('submit', handleSubmit);
downloadBtn.addEventListener('click', downloadEditedContent);
refreshHistoryBtn.addEventListener('click', () => loadHistory());
registerShortcuts();
loadHistory();
