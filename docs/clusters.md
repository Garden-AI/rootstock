# Example Configs

View live environment configurations from clusters where Rootstock is currently deployed. The configurations below are fetched from the dashboard API and show the exact environments, dependencies, and checkpoints available on each cluster.

Use these as a starting point when setting up your own cluster.

<style>
#clusters-container h2 { scroll-margin-top: 80px; }
.rs-toggle { cursor: pointer; user-select: none; }
.rs-toggle::before {
  content: "▶";
  display: inline-block;
  margin-right: 0.5em;
  font-size: 0.7em;
  transition: transform 0.2s;
}
.rs-toggle.open::before { transform: rotate(90deg); }
.rs-content { display: none; margin-left: 1.2em; }
.rs-content.open { display: block; }
.rs-copy {
  border: none;
  background: none;
  cursor: pointer;
  opacity: 0.4;
  padding: 0 4px;
  margin-left: 4px;
  vertical-align: baseline;
  transition: opacity 0.15s;
  line-height: 1;
}
.rs-copy:hover { opacity: 1; }
.rs-copy svg { width: 12px; height: 12px; vertical-align: -1px; }
</style>

<div id="clusters-container">
  <p>Loading cluster configurations...</p>
</div>

<script>
const API_URL = 'https://garden-ai-prod--rootstock-admin-dashboard.modal.run/';

function toggle(el) {
  el.classList.toggle('open');
  el.nextElementSibling.classList.toggle('open');
}

const COPY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
const CHECK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>';

function copyText(text, btn, e) {
  if (e) { e.stopPropagation(); e.preventDefault(); }
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = CHECK_ICON;
    setTimeout(() => btn.innerHTML = COPY_ICON, 1500);
  });
}

function copyData(btn, e) {
  if (e) { e.stopPropagation(); e.preventDefault(); }
  const text = btn.getAttribute('data-copy');
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = CHECK_ICON;
    setTimeout(() => btn.innerHTML = COPY_ICON, 1500);
  });
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

function renderCluster(manifest) {
  const envCount = Object.keys(manifest.environments).length;
  const slug = slugify(manifest.cluster);

  const envSections = Object.entries(manifest.environments).map(([name, env]) => {
    const deps = Object.entries(env.dependencies || {})
      .filter(([pkg]) => pkg !== 'rootstock')
      .map(([pkg, ver]) => `${pkg}==${ver}`)
      .join('\n');

    const checkpointList = (env.checkpoints && env.checkpoints.length > 0)
      ? `<p><strong>Checkpoints:</strong> ${env.checkpoints.map(c => `<code>${escapeHtml(c)}</code>`).join(', ')}</p>`
      : '';

    return `
      <div class="rs-toggle" onclick="toggle(this)"><strong>${escapeHtml(name)}</strong></div>
      <div class="rs-content">
        <p><strong>Python:</strong> ${escapeHtml(env.python_requires)} · <strong>Built:</strong> ${formatDate(env.built_at)}</p>
        ${checkpointList}
        ${env.source ? `
        <div class="rs-toggle" onclick="toggle(this)"><span>Source Code</span><button class="rs-copy" data-copy="${escapeHtml(env.source).replace(/"/g, '&quot;')}" onclick="copyData(this, event)">${COPY_ICON}</button></div>
        <div class="rs-content">
          <pre><code class="language-python">${escapeHtml(env.source)}</code></pre>
        </div>
        ` : ''}
        ${deps ? `
        <div class="rs-toggle" onclick="toggle(this)"><span>Dependencies (${Object.keys(env.dependencies || {}).filter(p => p !== 'rootstock').length})</span><button class="rs-copy" data-copy="${escapeHtml(deps).replace(/"/g, '&quot;')}" onclick="copyData(this, event)">${COPY_ICON}</button></div>
        <div class="rs-content">
          <pre><code>${escapeHtml(deps)}</code></pre>
        </div>
        ` : ''}
      </div>
    `;
  }).join('');

  return `
    <h2 id="${slug}">${escapeHtml(manifest.cluster)}</h2>
    <p>
      <strong>Rootstock:</strong> v${escapeHtml(manifest.rootstock_version)} ·
      <strong>Python:</strong> ${escapeHtml(manifest.python_version)} ·
      <strong>Environments:</strong> ${envCount}
    </p>
    <p>
      <strong>Root:</strong> <code>${escapeHtml(manifest.root)}</code><button class="rs-copy" onclick="copyText('${escapeHtml(manifest.root)}', this, event)">${COPY_ICON}</button><br>
      <strong>Maintainer:</strong> <a href="mailto:${escapeHtml(manifest.maintainer.email)}">${escapeHtml(manifest.maintainer.name)}</a><br>
      <strong>Updated:</strong> ${formatDate(manifest.last_updated)}
    </p>
    <h3>Environments</h3>
    ${envSections}
  `;
}

async function loadManifests() {
  const container = document.getElementById('clusters-container');
  try {
    const response = await fetch(API_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const manifests = data.manifests || data;

    if (!Array.isArray(manifests) || manifests.length === 0) {
      container.innerHTML = '<p>No cluster configurations available.</p>';
      return;
    }

    // Populate the TOC sidebar
    const tocNav = document.querySelector('.md-nav--secondary');
    if (tocNav) {
      const tocItems = manifests.map(m =>
        `<li class="md-nav__item">
          <a href="#${slugify(m.cluster)}" class="md-nav__link" onclick="event.preventDefault(); document.getElementById('${slugify(m.cluster)}').scrollIntoView({behavior: 'smooth'})">${escapeHtml(m.cluster)}</a>
        </li>`
      ).join('');
      tocNav.innerHTML = `
        <label class="md-nav__title" for="__toc">On this page</label>
        <ul class="md-nav__list" data-md-scrollfix>${tocItems}</ul>
      `;
    }

    container.innerHTML = manifests.map(renderCluster).join('<hr>');
  } catch (err) {
    console.error('Failed to load manifests:', err);
    container.innerHTML = `<p style="color: red;">Failed to load configurations: ${err.message}</p>`;
  }
}

loadManifests();
</script>
