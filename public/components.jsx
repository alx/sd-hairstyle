// components.jsx — UI components for the hairstyle studio

const { useState, useEffect, useRef, useCallback, useMemo } = React;

// ── Icons (tiny inline SVGs, stroke-based) ──────────────────────────────────
const Icon = {
  camera: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M3 8a2 2 0 0 1 2-2h2.5l1.2-1.6a1 1 0 0 1 .8-.4h5a1 1 0 0 1 .8.4L16.5 6H19a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8Z" fill="none" stroke="currentColor" strokeWidth="1.5"/><circle cx="12" cy="13" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.5"/></svg>),
  upload: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  refresh: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M3 12a9 9 0 0 1 15.5-6.3M21 4v5h-5M21 12a9 9 0 0 1-15.5 6.3M3 20v-5h5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  sparkle: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M12 3v6m0 6v6M3 12h6m6 0h6M6 6l3 3m6 6 3 3M18 6l-3 3M6 18l3-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
  heart: ({ filled, ...p }) => (<svg viewBox="0 0 24 24" {...p}><path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10Z" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/></svg>),
  share: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M12 4v12M7 9l5-5 5 5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  download: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M12 4v12M7 11l5 5 5-5M4 20h16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  calendar: (p) => (<svg viewBox="0 0 24 24" {...p}><rect x="3.5" y="5" width="17" height="16" rx="2" fill="none" stroke="currentColor" strokeWidth="1.5"/><path d="M8 3v4M16 3v4M3.5 10h17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
  close: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
  check: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M5 12l5 5L20 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  scissors: (p) => (<svg viewBox="0 0 24 24" {...p}><circle cx="6" cy="7" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.5"/><circle cx="6" cy="17" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.5"/><path d="m8 8.5 12 8.5M8 15.5 20 7" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
  arrow: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M5 12h14m-5-5 5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  menu: (p) => (<svg viewBox="0 0 24 24" {...p}><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
};

// ── Top navigation bar ──────────────────────────────────────────────────────
function TopBar({ view, setView, lookbookCount }) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <Icon.scissors width="18" height="18" />
        </span>
        <span className="brand-name">Maison Coupe</span>
        <span className="brand-tag">studio</span>
      </div>
      <nav className="nav">
        <button className={`nav-link ${view==='studio'?'is-active':''}`} onClick={() => setView('studio')}>Studio</button>
        <button className={`nav-link ${view==='lookbook'?'is-active':''}`} onClick={() => setView('lookbook')}>
          Lookbook
          {lookbookCount > 0 && <span className="nav-count">{lookbookCount}</span>}
        </button>
      </nav>
      <div className="topbar-right" />
    </header>
  );
}

// ── Selfie capture / preview / processing / result panel ────────────────────
function SelfiePanel({ state, setState, selfie, setSelfie, resultStyle, resultUrl, apiError, onClearStyle }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileRef = useRef(null);
  const [camErr, setCamErr] = useState(null);
  const [processingPct, setProcessingPct] = useState(0);

  // ── Webcam lifecycle ──────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCamErr(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      streamRef.current = stream;
      setState('camera'); // mounts <video ref={videoRef}>; stream attached by useEffect below
    } catch (e) {
      setCamErr(e.name === 'NotAllowedError'
        ? 'Camera access denied — you can still upload a photo below.'
        : 'No camera detected — upload a photo instead.');
    }
  }, [setState]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Attach stream after React commits the camera state (videoRef.current is now in the DOM)
  useEffect(() => {
    if (state !== 'camera' || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
    videoRef.current.play().catch(() => {});
  }, [state]);

  // ── Capture from video ────────────────────────────────────────────────
  const snap = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const canvas = document.createElement('canvas');
    const w = v.videoWidth || 720;
    const h = v.videoHeight || 960;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.translate(w, 0); ctx.scale(-1, 1); // mirror to match preview
    ctx.drawImage(v, 0, 0, w, h);
    setSelfie(canvas.toDataURL('image/jpeg', 0.92));
    stopCamera();
    setState('preview');
  }, [setSelfie, setState, stopCamera]);

  // ── File upload ───────────────────────────────────────────────────────
  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => { setSelfie(r.result); stopCamera(); setState('preview'); };
    r.readAsDataURL(f);
  };

  // ── Processing progress animation ─────────────────────────────────────
  // Animates the bar toward 99% while the real fetch runs in app.jsx.
  // The actual transition to 'result' is triggered by the fetch resolving.
  useEffect(() => {
    if (state !== 'processing') return;
    setProcessingPct(0);
    const start = Date.now();
    const DURATION = 30000;
    const id = setInterval(() => {
      const t = Math.min((Date.now() - start) / DURATION, 1);
      // ease-out: fast start, decelerates as it approaches 99% at ~37s
      setProcessingPct(99 * (1 - Math.pow(1 - t, 1.5)));
      if (t >= 1) clearInterval(id);
    }, 90);
    return () => clearInterval(id);
  }, [state]);

  // ── Render: state-specific surface ────────────────────────────────────
  const empty = state === 'empty';
  const showCamera = state === 'camera';
  const showPreview = state === 'preview';
  const showProcessing = state === 'processing';
  const showResult = state === 'result';

  return (
    <section className="selfie-panel">
      <div className="selfie-frame" data-state={state}>
        {empty && (
          <div className="selfie-empty">
            <div className="empty-glyph">
              <svg viewBox="0 0 80 80" width="64" height="64" aria-hidden="true">
                <circle cx="40" cy="30" r="14" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <path d="M14 70c4-14 14-20 26-20s22 6 26 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <h2 className="empty-title">Let's see your face.</h2>
            <p className="empty-sub">Center your head in the frame, with even light. We'll try cuts on you instantly.</p>
            <div className="empty-actions">
              <button className="btn btn-primary" onClick={startCamera}>
                <Icon.camera width="16" height="16" /><span>Use my camera</span>
              </button>
              <button className="btn btn-ghost" onClick={() => fileRef.current?.click()}>
                <Icon.upload width="16" height="16" /><span>Upload a photo</span>
              </button>
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
            </div>
            {camErr && <p className="empty-err">{camErr}</p>}
            <p className="empty-fine">Your photo is only sent to the local inference server — never stored externally.</p>
          </div>
        )}

        {showCamera && (
          <div className="cam-stage">
            <video ref={videoRef} className="cam-video" playsInline muted />
            <div className="cam-guides" aria-hidden="true">
              <div className="cam-oval" />
              <div className="cam-grid" />
            </div>
            <div className="cam-toolbar">
              <button className="cam-secondary" onClick={() => { stopCamera(); setState('empty'); }}>
                <Icon.close width="16" height="16" /><span>Cancel</span>
              </button>
              <button className="cam-shutter" onClick={snap} aria-label="Take photo">
                <span className="cam-shutter-ring" />
                <span className="cam-shutter-dot" />
              </button>
              <button className="cam-secondary" onClick={() => fileRef.current?.click()}>
                <Icon.upload width="16" height="16" /><span>Upload</span>
                <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
              </button>
            </div>
          </div>
        )}

        {(showPreview || showProcessing) && selfie && (
          <div className="preview-stage">
            <img src={selfie} alt="Your selfie" className="preview-img" />
            {showProcessing && (
              <div className="processing-overlay">
                <div className="processing-scan" />
                <div className="processing-card">
                  <div className="proc-spinner" aria-hidden="true" />
                  <div className="proc-text">
                    <div className="proc-label">Styling your look</div>
                    <div className="proc-sub">{processingStage(processingPct)}</div>
                  </div>
                  <div className="proc-bar"><div className="proc-bar-fill" style={{ width: `${processingPct}%` }} /></div>
                </div>
              </div>
            )}
            {showPreview && (
              <div className="preview-toolbar">
                <button className="btn btn-ghost btn-sm" onClick={() => { setSelfie(null); setState('empty'); }}>
                  <Icon.refresh width="14" height="14" /><span>Retake</span>
                </button>
                <div className="preview-hint">
                  <Icon.arrow width="16" height="16" />
                  <span>Now pick a haircut on the right →</span>
                </div>
              </div>
            )}
          </div>
        )}

        {showResult && selfie && resultStyle && (
          <BeforeAfter selfie={selfie} style={resultStyle} resultUrl={resultUrl} onClear={onClearStyle} />
        )}
      </div>
      {apiError && (
        <div className="api-error" role="alert">
          <span>{apiError}</span>
          <button onClick={() => onClearStyle()} aria-label="Dismiss error">✕</button>
        </div>
      )}
    </section>
  );
}

function processingStage(pct) {
  if (pct < 25) return 'Reading face geometry…';
  if (pct < 55) return 'Mapping hair region…';
  if (pct < 80) return 'Generating cut…';
  if (pct < 99) return 'Adding final polish…';
  return 'Ready';
}

// ── Before/After slider for the result view ─────────────────────────────────
function BeforeAfter({ selfie, style, resultUrl, onClear }) {
  const [pos, setPos] = useState(55);
  const wrapRef = useRef(null);
  const drag = (e) => {
    const w = wrapRef.current;
    if (!w) return;
    const r = w.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX ?? 0;
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  const onDown = (e) => {
    drag(e);
    const move = (ev) => drag(ev);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <div ref={wrapRef} className="ba-stage" onPointerDown={onDown}>
      {/* "Before" — your original selfie, on the left */}
      <img src={selfie} alt="Before" className="ba-img ba-before" />
      {/* "After" — SD-generated result if available, otherwise SVG overlay preview */}
      <div className="ba-after-clip" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        {resultUrl
          ? <img src={resultUrl} alt="After" className="ba-img ba-after" />
          : <>
              <img src={selfie} alt="After" className="ba-img ba-after" />
              <div className="ba-hair-overlay" data-style={style.id}><HairOverlay style={style} /></div>
            </>
        }
        <div className="ba-after-badge">
          <Icon.sparkle width="12" height="12" />
          <span>{style.name}</span>
        </div>
      </div>
      <div className="ba-handle" style={{ left: `${pos}%` }}>
        <div className="ba-handle-line" />
        <div className="ba-handle-knob">
          <svg viewBox="0 0 24 24" width="20" height="20"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </div>
      </div>
      <div className="ba-label ba-label-l">Before</div>
      <div className="ba-label ba-label-r">After</div>
      <button className="ba-clear" onClick={onClear} aria-label="Try a different cut">
        <Icon.refresh width="14" height="14" /><span>Try another</span>
      </button>
    </div>
  );
}

// Simulated "after" — we tint a stylized hair shape onto the photo. It's a
// stand-in for the real ML overlay, but the shape changes per cut so the
// user sees real differentiation between styles.
function HairOverlay({ style }) {
  // The actual photo is freeform — we can't match its exact head position.
  // We use an SVG that occupies the top-center of the frame, sized for a
  // typical centered headshot, with mood-driven shapes.
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" className="ha-svg" aria-hidden="true">
      <defs>
        <linearGradient id={`hair-${style.id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--hair-hi)" />
          <stop offset="100%" stopColor="var(--hair-lo)" />
        </linearGradient>
        <filter id={`blur-${style.id}`}><feGaussianBlur stdDeviation="0.35" /></filter>
      </defs>
      <g fill={`url(#hair-${style.id})`} filter={`url(#blur-${style.id})`} opacity="0.92">
        {hairOverlayPath(style.id)}
      </g>
    </svg>
  );
}

function hairOverlayPath(id) {
  const paths = {
    bob:      <path d="M28 22 Q28 8 50 7 Q72 8 72 22 L74 44 Q74 38 66 36 L64 22 Q50 18 36 22 L34 36 Q26 38 26 44 Z" />,
    sidepart: <path d="M30 22 Q30 8 52 7 Q72 8 72 22 Q70 16 56 14 L52 20 Q42 16 36 20 Q30 18 30 22 Z" />,
    pixie:    <path d="M32 18 Q32 8 50 7 Q70 8 70 18 Q70 22 64 22 L62 16 Q50 12 38 16 Q34 16 32 18 Z" />,
    lob:      <path d="M26 22 Q26 8 50 7 Q74 8 74 22 L76 60 L68 60 L68 38 L64 18 Q50 14 36 18 L32 38 L32 60 L24 60 Z" />,
    mullet:   <path d="M30 18 Q30 6 50 6 Q70 6 70 18 Q70 18 64 18 L62 14 Q50 10 38 14 L36 18 Q30 18 30 18 Z M68 26 L72 64 L62 64 L64 32 Z M32 26 L28 64 L38 64 L36 32 Z" />,
    fade:     <path d="M36 16 Q36 6 50 6 Q64 6 64 16 Q64 20 60 20 L58 12 Q50 8 42 12 L40 20 Q36 20 36 16 Z" />,
    asym:     <path d="M26 18 Q26 8 50 7 Q74 8 74 18 L78 56 L66 56 L64 28 L60 14 Q50 10 36 14 L34 28 L30 36 L28 28 L26 22 Z" />,
    wolf:     <path d="M24 22 Q24 6 50 5 Q76 6 76 22 L80 66 L66 66 L64 40 L60 18 Q50 14 36 18 L32 40 L30 66 L20 66 Z" />,
    slick:    <path d="M32 18 Q32 6 50 6 Q70 6 70 20 Q72 32 66 38 L62 18 Q50 14 38 18 L34 38 Q28 32 32 18 Z" />,
    crop:     <path d="M34 16 Q34 6 50 6 Q66 6 66 16 Q66 22 60 22 L58 14 Q50 10 42 14 L40 22 Q34 22 34 16 Z" />,
    curtain:  <path d="M28 22 Q28 8 50 7 Q72 8 72 22 L72 52 L66 52 L64 32 Q56 18 50 18 Q44 18 36 32 L34 52 L28 52 Z M42 14 Q47 22 50 22 Q53 22 58 14" />,
    blunt:    <path d="M28 24 Q28 8 50 7 Q72 8 72 24 L72 62 L64 62 L62 18 Q50 14 38 18 L36 62 L28 62 Z" />,
  };
  return paths[id] || paths.bob;
}

// ── Style gallery ───────────────────────────────────────────────────────────
function StyleGallery({ haircuts, selectedId, setSelected, favorites, toggleFav, onTry, canTry }) {
  const [mood, setMood] = useState('all');
  const filtered = haircuts.filter((h) => mood === 'all' || h.mood === mood);

  return (
    <section className="gallery-panel">
      <header className="gallery-head">
        <div>
          <h2 className="gallery-title">The catalog</h2>
          <p className="gallery-sub">12 hand-picked cuts from our master stylists.</p>
        </div>
        <div className="gallery-filters" role="tablist">
          {MOODS.map((m) => (
            <button key={m.id} className={`chip ${mood===m.id?'is-active':''}`}
                    onClick={() => setMood(m.id)} role="tab" aria-selected={mood===m.id}>
              {m.label}
            </button>
          ))}
        </div>
      </header>

      <div className="gallery-grid">
        {filtered.map((h) => {
          const isSel = h.id === selectedId;
          const isFav = favorites.includes(h.id);
          return (
            <article key={h.id} className={`cut-card ${isSel?'is-selected':''}`}
                     onClick={() => setSelected(h.id)}>
              <div className="cut-thumb">
                <img src={h.img} alt={h.name}
                     onError={(e) => { e.currentTarget.style.display='none'; e.currentTarget.nextSibling.style.display=''; }}
                     style={{ width:'100%', height:'100%', objectFit:'cover', borderRadius:'inherit', display:'block' }} />
                <span style={{ display:'none' }}><HairSilhouette id={h.id} /></span>
                <button className={`fav-btn ${isFav?'is-fav':''}`}
                        onClick={(e) => { e.stopPropagation(); toggleFav(h.id); }}
                        aria-label={isFav ? 'Remove from lookbook' : 'Save to lookbook'}>
                  <Icon.heart width="15" height="15" filled={isFav} />
                </button>
                <div className="cut-mood">{h.mood}</div>
              </div>
              <div className="cut-meta">
                <div className="cut-name">{h.name}</div>
                <div className="cut-spec">
                  <span>{h.length}</span>
                </div>
              </div>
              {isSel && (
                <div className="cut-actions">
                  <p className="cut-desc">{h.desc}</p>
                  <button className="btn btn-primary btn-sm" disabled={!canTry}
                          onClick={(e) => { e.stopPropagation(); onTry(h); }}>
                    <Icon.sparkle width="14" height="14" />
                    <span>{canTry ? 'Try this on me' : 'Take a selfie first'}</span>
                  </button>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

// ── Lookbook drawer ─────────────────────────────────────────────────────────
function Lookbook({ haircuts, favorites, toggleFav, onTry, onClose, canTry }) {
  const items = haircuts.filter((h) => favorites.includes(h.id));
  return (
    <div className="drawer">
      <header className="drawer-head">
        <div>
          <h2 className="drawer-title">My Lookbook</h2>
          <p className="drawer-sub">{items.length} saved {items.length === 1 ? 'look' : 'looks'}</p>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close lookbook">
          <Icon.close width="18" height="18" />
        </button>
      </header>
      {items.length === 0 ? (
        <div className="drawer-empty">
          <div className="empty-glyph"><Icon.heart width="40" height="40" /></div>
          <h3>No saves yet</h3>
          <p>Tap the heart on any cut to save it here for later.</p>
        </div>
      ) : (
        <div className="lookbook-grid">
          {items.map((h) => (
            <article key={h.id} className="lookbook-card">
              <div className="lookbook-thumb">
                <img src={h.img} alt={h.name}
                     onError={(e) => { e.currentTarget.style.display='none'; e.currentTarget.nextSibling.style.display=''; }}
                     style={{ width:'100%', height:'100%', objectFit:'cover', borderRadius:'inherit', display:'block' }} />
                <span style={{ display:'none' }}><HairSilhouette id={h.id} /></span>
              </div>
              <div className="lookbook-body">
                <div>
                  <div className="cut-name">{h.name}</div>
                  <div className="cut-spec">
                    <span>{h.length}</span><span className="dot" /><span style={{ textTransform: 'capitalize' }}>{h.mood}</span>
                  </div>
                </div>
                <div className="lookbook-actions">
                  <button className="btn btn-ghost btn-sm" onClick={() => toggleFav(h.id)}>Remove</button>
                  <button className="btn btn-primary btn-sm" disabled={!canTry} onClick={() => onTry(h)}>
                    Try on
                  </button>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Share modal ─────────────────────────────────────────────────────────────
function ShareModal({ style, selfie, resultUrl, onClose }) {
  const [copied, setCopied] = useState(false);
  const shareUrl = `maisoncoupe.studio/look/${style.id}-${Math.random().toString(36).slice(2,7)}`;
  const previewSrc = resultUrl || selfie;
  const onCopy = () => {
    navigator.clipboard?.writeText(`https://${shareUrl}`).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  const onDownload = () => {
    if (!previewSrc) return;
    const a = document.createElement('a');
    a.href = previewSrc;
    a.download = `maison-coupe-${style.id}.jpg`;
    a.click();
  };
  return (
    <Modal onClose={onClose}>
      <div className="share-modal">
        <header className="modal-head">
          <h2>Share your new look</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <Icon.close width="18" height="18" />
          </button>
        </header>
        <div className="share-preview">
          {previewSrc && <img src={previewSrc} alt="" className="share-img" />}
          <div className="share-meta">
            <span className="share-eyebrow">trying on</span>
            <h3>{style.name}</h3>
            <p>{style.desc}</p>
          </div>
        </div>
        <div className="share-link">
          <div className="share-url">{shareUrl}</div>
          <button className="btn btn-ghost btn-sm" onClick={onCopy}>
            {copied ? <><Icon.check width="14" height="14" /><span>Copied</span></> : <span>Copy link</span>}
          </button>
        </div>
        <div className="share-row">
          {['Instagram', 'WhatsApp', 'Messages', 'Email'].map((s) => (
            <button key={s} className="share-chip">{s}</button>
          ))}
        </div>
        <button className="btn btn-primary btn-block" onClick={onDownload}>
          <Icon.download width="16" height="16" />
          <span>Download image</span>
        </button>
      </div>
    </Modal>
  );
}


// ── Modal shell ─────────────────────────────────────────────────────────────
function Modal({ onClose, wide, children }) {
  useEffect(() => {
    const k = (e) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', k);
    return () => document.removeEventListener('keydown', k);
  }, [onClose]);
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className={`modal ${wide?'modal-wide':''}`} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

// ── Result action bar (shown when a style is locked in) ─────────────────────
function ResultBar({ style, isFav, toggleFav, onShare }) {
  return (
    <div className="resultbar">
      <div className="resultbar-meta">
        <span className="resultbar-eyebrow">Now wearing</span>
        <div className="resultbar-name">{style.name}</div>
      </div>
      <div className="resultbar-actions">
        <button className={`btn btn-ghost ${isFav?'is-fav':''}`} onClick={() => toggleFav(style.id)}>
          <Icon.heart width="15" height="15" filled={isFav} />
          <span>{isFav ? 'Saved' : 'Save'}</span>
        </button>
        <button className="btn btn-primary" onClick={onShare}>
          <Icon.share width="15" height="15" /><span>Share look</span>
        </button>
      </div>
    </div>
  );
}

Object.assign(window, {
  Icon, TopBar, SelfiePanel, BeforeAfter, StyleGallery, Lookbook,
  ShareModal, Modal, ResultBar,
});
