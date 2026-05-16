const thread     = document.getElementById('chat-thread');
const input      = document.getElementById('question-input');
const sendBtn    = document.getElementById('send-btn');
const emptyState = document.getElementById('empty-state');

let sessionId = crypto.randomUUID();

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

function fillQuestion(btn) {
  input.value = btn.textContent.trim();
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  input.focus();
}

async function clearChat() {
  try {
    await fetch('/clear', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId }),
    });
  } catch { /* silent fail */ }

  thread.querySelectorAll('.msg-row').forEach(m => m.remove());
  emptyState.style.display = '';
  sessionId = crypto.randomUUID();
}

function formatText(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^\* (.+)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul style="margin:8px 0 0 16px;padding:0">$1</ul>')
    .replace(/\n/g, '<br>');
}

function toggleSources(btn) {
  const list = btn.nextElementSibling;
  const icon = btn.querySelector('.toggle-icon');
  const open = list.style.display === 'none';
  list.style.display = open ? 'flex' : 'none';
  if (open) list.style.flexDirection = 'column';
  icon.classList.toggle('open', open);
  btn.querySelector('.toggle-label').textContent = open
    ? 'Hide sources'
    : `${list.children.length} source${list.children.length > 1 ? 's' : ''}`;
}

function buildSources(sources) {
  if (!sources || sources.length === 0) return '';
  const chips = sources.map(s => `
    <div class="source-chip">
      <div class="source-dot"></div>
      <span class="source-title">${s.title || s.filename.replace('.md','').replace(/-/g,' ')}</span>
      <span class="source-score">${Math.round(s.score * 100)}% match</span>
    </div>
  `).join('');

  return `
    <div class="sources-wrap">
      <button class="sources-toggle" onclick="toggleSources(this)">
        <i class="bi bi-chevron-right toggle-icon"></i>
        <span class="toggle-label">${sources.length} source${sources.length > 1 ? 's' : ''}</span>
      </button>
      <div class="sources-list" style="display:none">${chips}</div>
    </div>
  `;
}

function buildFeedback(question, answer) {
  const uid = Math.random().toString(36).slice(2, 8);
  return `
    <div class="feedback-wrap" id="fb-${uid}">
      <span class="feedback-label">Was this helpful?</span>
      <button class="feedback-btn"
        id="up-${uid}"
        data-uid="${uid}"
        data-vote="up"
        data-question="${encodeURIComponent(question)}"
        data-answer="${encodeURIComponent(answer)}"
        onclick="sendFeedback(this)"
        title="Helpful">
        <i class="bi bi-hand-thumbs-up"></i> Yes
      </button>
      <button class="feedback-btn"
        id="down-${uid}"
        data-uid="${uid}"
        data-vote="down"
        data-question="${encodeURIComponent(question)}"
        data-answer="${encodeURIComponent(answer)}"
        onclick="sendFeedback(this)"
        title="Not helpful">
        <i class="bi bi-hand-thumbs-down"></i> No
      </button>
    </div>
  `;
}


async function sendFeedback(btn) {
  const uid      = btn.dataset.uid;
  const vote     = btn.dataset.vote;
  const question = decodeURIComponent(btn.dataset.question);
  const answer   = decodeURIComponent(btn.dataset.answer);

  const upBtn   = document.getElementById(`up-${uid}`);
  const downBtn = document.getElementById(`down-${uid}`);
  const wrap    = document.getElementById(`fb-${uid}`);

  upBtn.disabled   = true;
  downBtn.disabled = true;

  if (vote === 'up') {
    upBtn.classList.add('voted-up');
    wrap.querySelector('.feedback-label').textContent = 'Thanks for the feedback!';
  } else {
    downBtn.classList.add('voted-down');
    wrap.querySelector('.feedback-label').textContent = 'Thanks — we will improve!';
  }

  try {
    await fetch('/feedback', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId, question, answer, vote }),
    });
  } catch { /* silent fail */ }
}


function appendMessage(role, content, sources = [], question = '') {
  if (emptyState) emptyState.style.display = 'none';

  const isUser = role === 'user';
  const row    = document.createElement('div');
  row.className = `msg-row ${role}`;

  row.innerHTML = `
    <div class="avatar ${role}">${isUser ? 'You' : 'H'}</div>
    <div class="msg-content">
      <div class="bubble ${role}">${formatText(content)}</div>
      ${isUser ? '' : buildSources(sources)}
      ${isUser ? '' : buildFeedback(question, content)}
    </div>
  `;

  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
}

function showTyping() {
  if (emptyState) emptyState.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="avatar bot">H</div>
    <div class="msg-content">
      <div class="typing-bubble">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}

async function sendQuestion() {
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  appendMessage('user', question);
  showTyping();

  try {
    const res  = await fetch('/ask', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question, session_id: sessionId }),
    });
    const data = await res.json();
    removeTyping();

    if (data.error) {
      appendMessage('bot', `Something went wrong: ${data.error}`);
    } else {
      if (data.session_id) sessionId = data.session_id;
      appendMessage('bot', data.answer, data.sources, question);
    }

  } catch {
    removeTyping();
    appendMessage('bot', 'Could not reach the server. Please try again.');
  }

  sendBtn.disabled = false;
  input.focus();
}