// app.jsx — top-level orchestrator and Tweaks wiring

const { useState, useEffect, useMemo } = React;

// ── API helpers ────────────────────────────────────────────────

function dataUrlToBlob(dataUrl) {
  const [header, b64] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

async function silhouetteToBlob(id) {
  const d = HAIR_SVG_PATHS[id] || HAIR_SVG_PATHS.bob;
  const svgStr = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
    <rect width="120" height="120" fill="#e8c8a8"/>
    <path d="${d}" fill="#2a1f17"/>
  </svg>`;
  const blob = new Blob([svgStr], { type: 'image/svg+xml' });
  const url  = URL.createObjectURL(blob);
  const img  = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  const canvas = Object.assign(document.createElement('canvas'), { width: 120, height: 120 });
  canvas.getContext('2d').drawImage(img, 0, 0);
  URL.revokeObjectURL(url);
  return new Promise((res) => canvas.toBlob(res, 'image/png'));
}

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "aesthetic": "editorial",
  "density": "regular"
}/*EDITMODE-END*/;

const AESTHETICS = {
  editorial: {
    label: 'Editorial',
    vars: {
      '--bg':            '#f6f1ea',
      '--bg-soft':       '#ede4d6',
      '--surface':       '#ffffff',
      '--ink':           '#2a1f17',
      '--ink-soft':      '#6b5a4c',
      '--ink-faint':     '#a89784',
      '--line':          '#e0d3c0',
      '--line-strong':   '#cdb89e',
      '--accent':        '#c4592c',
      '--accent-soft':   '#f1d8c8',
      '--accent-ink':    '#ffffff',
      '--hair-hi':       '#3d2a1f',
      '--hair-lo':       '#1a0f08',
      '--font-display':  '"Cormorant Garamond", "Playfair Display", Georgia, serif',
      '--font-body':     '"Inter Tight", "Inter", system-ui, sans-serif',
      '--font-mono':     '"JetBrains Mono", ui-monospace, monospace',
      '--radius':        '14px',
      '--radius-sm':     '8px',
      '--shadow':        '0 1px 0 rgba(255,255,255,.6) inset, 0 12px 36px -16px rgba(67,40,20,.18)',
    },
    titleWeight: 500,
    titleItalic: true,
  },
  pastel: {
    label: 'Pastel',
    vars: {
      '--bg':            '#fbeae5',
      '--bg-soft':       '#f5d8d0',
      '--surface':       '#fffaf7',
      '--ink':           '#3c1f2a',
      '--ink-soft':      '#7a5664',
      '--ink-faint':     '#b89aa6',
      '--line':          '#f0d4cd',
      '--line-strong':   '#dcb5ab',
      '--accent':        '#8e3a5a',
      '--accent-soft':   '#f4cfdc',
      '--accent-ink':    '#ffffff',
      '--hair-hi':       '#5a2e3e',
      '--hair-lo':       '#2c1620',
      '--font-display':  '"Fraunces", Georgia, serif',
      '--font-body':     '"Manrope", system-ui, sans-serif',
      '--font-mono':     '"JetBrains Mono", ui-monospace, monospace',
      '--radius':        '20px',
      '--radius-sm':     '12px',
      '--shadow':        '0 1px 0 rgba(255,255,255,.7) inset, 0 16px 40px -18px rgba(120,40,70,.22)',
    },
    titleWeight: 600,
    titleItalic: false,
  },
  clinical: {
    label: 'Clinical',
    vars: {
      '--bg':            '#f3f5f7',
      '--bg-soft':       '#e6eaef',
      '--surface':       '#ffffff',
      '--ink':           '#0d1620',
      '--ink-soft':      '#54616f',
      '--ink-faint':     '#94a0ae',
      '--line':          '#d9e0e8',
      '--line-strong':   '#b9c4d0',
      '--accent':        '#2c5cd6',
      '--accent-soft':   '#dee7fb',
      '--accent-ink':    '#ffffff',
      '--hair-hi':       '#1a2230',
      '--hair-lo':       '#0a0f17',
      '--font-display':  '"Söhne", "Inter Tight", system-ui, sans-serif',
      '--font-body':     '"Inter", system-ui, sans-serif',
      '--font-mono':     '"JetBrains Mono", ui-monospace, monospace',
      '--radius':        '6px',
      '--radius-sm':     '4px',
      '--shadow':        '0 0 0 .5px rgba(13,22,32,.06), 0 1px 2px rgba(13,22,32,.04)',
    },
    titleWeight: 600,
    titleItalic: false,
  },
  fashion: {
    label: 'Fashion',
    vars: {
      '--bg':            '#0a0a0a',
      '--bg-soft':       '#161616',
      '--surface':       '#1c1c1c',
      '--ink':           '#fafafa',
      '--ink-soft':      '#a0a0a0',
      '--ink-faint':     '#6a6a6a',
      '--line':          '#2a2a2a',
      '--line-strong':   '#3c3c3c',
      '--accent':        '#e63946',
      '--accent-soft':   '#3a1418',
      '--accent-ink':    '#ffffff',
      '--hair-hi':       '#1a1a1a',
      '--hair-lo':       '#000000',
      '--font-display':  '"Bodoni Moda", "Playfair Display", Georgia, serif',
      '--font-body':     '"Inter", system-ui, sans-serif',
      '--font-mono':     '"JetBrains Mono", ui-monospace, monospace',
      '--radius':        '2px',
      '--radius-sm':     '2px',
      '--shadow':        'none',
    },
    titleWeight: 700,
    titleItalic: false,
  },
};

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const aes = AESTHETICS[t.aesthetic] || AESTHETICS.editorial;

  // Apply theme vars to <html> so the whole document inherits.
  useEffect(() => {
    const root = document.documentElement;
    Object.entries(aes.vars).forEach(([k, v]) => root.style.setProperty(k, v));
    root.dataset.aesthetic = t.aesthetic;
    root.dataset.density = t.density;
    root.style.setProperty('--title-weight', String(aes.titleWeight));
    root.style.setProperty('--title-style', aes.titleItalic ? 'italic' : 'normal');
  }, [t.aesthetic, t.density, aes]);

  // ── Persistent app state ─────────────────────────────────────────────
  const [view, setView] = useState('studio');           // 'studio' | 'lookbook'
  const [selfie, setSelfie] = useState(null);            // dataURL
  const [selfieState, setSelfieState] = useState('empty'); // empty | camera | preview | processing | result
  const [selectedId, setSelectedId] = useState(null);    // currently highlighted in gallery
  const [resultStyle, setResultStyle] = useState(null);  // style we processed onto the photo
  const [resultUrl, setResultUrl] = useState(null);      // ObjectURL of the JPEG returned by the API
  const [apiError, setApiError] = useState(null);        // error message string | null
  const [isDemoResult, setIsDemoResult] = useState(false); // true when showing pre-computed fallback
  const [favorites, setFavorites] = useState(() => {
    try { return JSON.parse(localStorage.getItem('mc.favs') || '[]'); }
    catch (e) { return []; }
  });
  const [shareOpen, setShareOpen] = useState(false);

  useEffect(() => {
    try { localStorage.setItem('mc.favs', JSON.stringify(favorites)); } catch (e) {}
  }, [favorites]);

  const toggleFav = (id) => setFavorites((f) => f.includes(id) ? f.filter((x) => x !== id) : [...f, id]);

  // canTry: allow clicking even without a selfie — demo fallback will load one
  const canTry = selfieState !== 'processing';

  const onTry = async (style) => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setSelectedId(style.id);
    setResultStyle(style);
    setApiError(null);
    setResultUrl((prev) => { if (prev && !isDemoResult) URL.revokeObjectURL(prev); return null; });
    if (view !== 'studio') setView('studio');

    // Capture the selfie value at call time — state updates won't change this closure
    const hasSelfie = !!selfie;

    // Pre-load demo selfie if visitor hasn't uploaded their own
    if (!hasSelfie) {
      try {
        const r = await fetch('static/demo/selfie.jpg');
        if (r.ok) {
          const blob = await r.blob();
          const dataUrl = await new Promise((res) => {
            const reader = new FileReader();
            reader.onload = (e) => res(e.target.result);
            reader.readAsDataURL(blob);
          });
          setSelfie(dataUrl);
        }
      } catch (_) {}
    }

    setSelfieState('processing');

    // Try the live API when the visitor uploaded their own selfie
    if (hasSelfie) {
      try {
        const form = new FormData();
        form.append('selfie',       dataUrlToBlob(selfie),            'selfie.jpg');
        form.append('hairstyle',    await silhouetteToBlob(style.id), 'hairstyle.png');
        form.append('hairstyle_id', style.id);
        form.append('prompt',       style.prompt || style.tags.join(', '));
        const strengthMap = { compact: '0.72', regular: '0.85', roomy: '0.93' };
        form.append('strength',  strengthMap[t.density] ?? '0.85');
        form.append('steps',     '8');
        form.append('cfg_scale', '3.5');
        form.append('seed',      '-1');

        const res = await fetch('/transfer-haircut', { method: 'POST', body: form });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          throw new Error(j.error || `Server error ${res.status}`);
        }
        const imgBlob = await res.blob();
        setResultUrl(URL.createObjectURL(imgBlob));
        setIsDemoResult(false);
        setSelfieState('result');
        return;
      } catch (_) {
        // API unavailable — fall through to demo images
      }
    }

    // Demo fallback: serve pre-computed images from static/demo/
    try {
      const demoRes = await fetch(`static/demo/${style.id}.jpg`, { method: 'HEAD' });
      if (demoRes.ok) {
        setResultUrl(`static/demo/${style.id}.jpg`);
        setIsDemoResult(true);
        setSelfieState('result');
        return;
      }
    } catch (_) {}

    // Nothing worked
    setApiError(hasSelfie
      ? 'Server unavailable and no demo image found. Try again later.'
      : 'No demo images found. Upload your photo to try a style.');
    setSelfieState(hasSelfie || selfie ? 'preview' : 'empty');
  };

  const onClearStyle = () => {
    setResultStyle(null);
    setResultUrl((prev) => { if (prev && !isDemoResult) URL.revokeObjectURL(prev); return null; });
    setIsDemoResult(false);
    setApiError(null);
    setSelfieState(selfie ? 'preview' : 'empty');
  };


  const onShare = () => {
    if (!resultStyle) return;
    setShareOpen(true);
  };

  return (
    <>
      <TopBar view={view} setView={setView} lookbookCount={favorites.length} />

      <main className="stage" data-view={view}>
        {view === 'studio' && (
          <div className="studio-grid">
            <div className="studio-left">
              <SelfiePanel
                state={selfieState}
                setState={setSelfieState}
                selfie={selfie}
                setSelfie={setSelfie}
                resultStyle={resultStyle}
                resultUrl={resultUrl}
                apiError={apiError}
                onClearStyle={onClearStyle}
              />
              {selfieState === 'result' && resultStyle && (
                <ResultBar
                  style={resultStyle}
                  isFav={favorites.includes(resultStyle.id)}
                  toggleFav={toggleFav}
                  onShare={onShare}
                  />
              )}
              {selfieState === 'result' && isDemoResult && (
                <div style={{
                  marginTop: '10px', padding: '10px 14px', borderRadius: 'var(--radius-sm)',
                  background: 'var(--accent-soft)', color: 'var(--accent)',
                  fontSize: '.8125rem', fontWeight: 500, textAlign: 'center',
                  lineHeight: 1.4,
                }}>
                  Demo preview — upload your photo for a personalised result
                </div>
              )}
            </div>
            <div className="studio-right">
              <StyleGallery
                haircuts={HAIRCUTS}
                selectedId={selectedId}
                setSelected={setSelectedId}
                favorites={favorites}
                toggleFav={toggleFav}
                onTry={onTry}
                canTry={canTry}
              />
            </div>
          </div>
        )}

        {view === 'lookbook' && (
          <Lookbook
            haircuts={HAIRCUTS}
            favorites={favorites}
            toggleFav={toggleFav}
            onTry={(s) => { setView('studio'); onTry(s); }}
            onClose={() => setView('studio')}
            canTry={canTry}
          />
        )}
      </main>

      {shareOpen && resultStyle && (
        <ShareModal style={resultStyle} selfie={selfie} resultUrl={resultUrl} onClose={() => setShareOpen(false)} />
      )}

      <TweaksPanel title="Tweaks">
        <TweakSection label="Aesthetic">
          <TweakSelect label="Direction" value={t.aesthetic}
                       options={Object.entries(AESTHETICS).map(([k, v]) => ({ value: k, label: v.label }))}
                       onChange={(v) => setTweak('aesthetic', v)} />
          <TweakRadio label="Density" value={t.density}
                      options={['compact', 'regular', 'roomy']}
                      onChange={(v) => setTweak('density', v)} />
        </TweakSection>
        <TweakSection label="State (debug)">
          <TweakButton label="Reset selfie" secondary onClick={() => {
            setSelfie(null); setSelfieState('empty'); setResultStyle(null);
            setSelectedId(null); setIsDemoResult(false); setResultUrl(null);
          }} />
          <TweakButton label="Clear lookbook" secondary onClick={() => setFavorites([])} />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
