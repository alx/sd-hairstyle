// data.jsx — haircut catalog and shared constants

const HAIRCUTS = [
  // ── Classic ──────────────────────────────────────────────
  { id: 'bob',         name: 'The French Bob',      mood: 'classic', length: 'Short',  duration: '45 min', price: 85,
    desc: 'Chin-length blunt cut with soft inward curve. Universally flattering.',
    tags: ['blunt', 'chin-length'],
    img: 'static/haircuts/bob.jpg',
    prompt: 'chin-length french bob haircut, blunt cut soft inward curve' },
  { id: 'sidepart',    name: 'Side Part Tailored',  mood: 'classic', length: 'Medium', duration: '40 min', price: 70,
    desc: 'Clean side-part with tapered sides. A boardroom mainstay.',
    tags: ['tapered', 'polished', 'professional'],
    img: 'static/haircuts/sidepart.jpg',
    prompt: 'classic side part haircut, tapered sides, polished professional' },
  { id: 'pixie',       name: 'Audrey Pixie',        mood: 'classic', length: 'Short',  duration: '50 min', price: 90,
    desc: 'Cropped close at the nape with longer fringe up top.',
    tags: ['cropped', 'fringe', 'classic'],
    img: 'static/haircuts/pixie.jpg',
    prompt: 'audrey pixie cut, cropped close at nape, longer fringe on top' },
  { id: 'lob',         name: 'Layered Lob',         mood: 'classic', length: 'Medium', duration: '55 min', price: 95,
    desc: 'Collarbone-grazing length with subtle face-framing layers.',
    tags: ['layered', 'soft', 'versatile'],
    img: 'static/haircuts/lob.jpg',
    prompt: 'layered lob, collarbone length hair, face-framing layers' },

  // ── Edgy ─────────────────────────────────────────────────
  { id: 'mullet',      name: 'Modern Mullet',       mood: 'edgy',    length: 'Medium', duration: '60 min', price: 105,
    desc: 'Short on top, long in back — reimagined with razor-cut texture.',
    tags: ['textured', 'statement', 'rocker'],
    img: 'static/haircuts/mullet.jpg',
    prompt: 'modern mullet, short on top long in back, razor-cut texture' },
  { id: 'fade',        name: 'Skin Fade + Pomp',    mood: 'edgy',    length: 'Short',  duration: '45 min', price: 75,
    desc: 'High skin fade transitioning into volume up top.',
    tags: ['fade', 'sharp', 'urban'],
    img: 'static/haircuts/fade.jpg',
    prompt: 'high skin fade with pompadour, volume on top, urban style' },
  { id: 'asym',        name: 'Asymmetric Shag',     mood: 'edgy',    length: 'Medium', duration: '65 min', price: 110,
    desc: 'Uneven layers with one side dramatically shorter.',
    tags: ['asymmetric', 'shag', 'bold'],
    img: 'static/haircuts/asym.jpg',
    prompt: 'asymmetric shag, one side dramatically shorter, bold layered haircut' },
  { id: 'wolf',        name: 'The Wolf Cut',        mood: 'edgy',    length: 'Long',   duration: '70 min', price: 120,
    desc: 'Wolfy mix of shag and mullet — heavy crown, wispy ends.',
    tags: ['wolf', 'textured', 'trending'],
    img: 'static/haircuts/wolf.jpg',
    prompt: 'wolf cut hairstyle, shag-mullet mix, heavy crown wispy ends' },

  // ── Minimal ──────────────────────────────────────────────
  { id: 'slick',       name: 'Slick Back',          mood: 'minimal', length: 'Medium', duration: '35 min', price: 65,
    desc: 'All length pulled straight back. Strong silhouette, zero fuss.',
    tags: ['slick', 'sleek', 'minimal'],
    img: 'static/haircuts/slick.jpg',
    prompt: 'slicked-back hair, all hair pulled straight back, strong silhouette' },
  { id: 'crop',        name: 'French Crop',         mood: 'minimal', length: 'Short',  duration: '40 min', price: 70,
    desc: 'Short, blunt fringe with cropped sides. Quiet confidence.',
    tags: ['crop', 'fringe', 'understated'],
    img: 'static/haircuts/crop.jpg',
    prompt: 'french crop haircut, short blunt fringe, cropped sides' },
  { id: 'curtain',     name: 'Curtain Bangs',       mood: 'minimal', length: 'Long',   duration: '50 min', price: 85,
    desc: 'Center-parted fringe that frames the cheekbones.',
    tags: ['bangs', 'soft', 'parisian'],
    img: 'static/haircuts/curtain.jpg',
    prompt: 'curtain bangs, center-parted fringe framing cheekbones' },
  { id: 'blunt',       name: 'Blunt One-Length',    mood: 'minimal', length: 'Long',   duration: '45 min', price: 80,
    desc: 'Single weight, no layers, mirror-shine ends.',
    tags: ['blunt', 'sleek', 'graphic'],
    img: 'static/haircuts/blunt.jpg',
    prompt: 'blunt one-length hair, no layers, mirror-shine ends' },
];

const MOODS = [
  { id: 'all',     label: 'All looks' },
  { id: 'classic', label: 'Classic' },
  { id: 'edgy',    label: 'Edgy' },
  { id: 'minimal', label: 'Minimal' },
];

// Shared path `d` strings for the 12 silhouettes — consumed by HairSilhouette
// and by the canvas renderer in app.jsx that generates the hairstyle reference
// image sent to the Flask /transfer-haircut endpoint.
const HAIR_SVG_PATHS = {
  bob:      "M30 70 Q30 38 60 36 Q90 38 90 70 L88 92 Q88 84 76 82 L74 60 Q60 56 46 60 L44 82 Q32 84 32 92 Z",
  sidepart: "M34 68 Q34 38 62 36 Q90 38 90 60 Q88 50 70 48 L66 56 Q54 52 44 60 Q36 58 34 68 Z",
  pixie:    "M38 60 Q38 38 60 36 Q86 38 86 58 Q86 64 80 66 Q78 56 70 54 L66 58 Q56 52 48 58 Q40 58 38 60 Z",
  lob:      "M30 70 Q30 38 60 36 Q90 38 90 68 L92 108 L84 108 L84 80 L80 56 Q60 52 40 56 L36 80 L36 108 L28 108 Z",
  mullet:   "M34 60 Q34 36 60 36 Q86 36 86 60 Q86 60 80 60 L78 56 Q60 52 42 56 L40 60 Q34 60 34 60 Z M84 70 L88 112 L78 112 L80 78 Z M36 70 L32 112 L42 112 L40 78 Z",
  fade:     "M40 56 Q40 36 60 36 Q80 36 80 56 Q80 60 76 60 L74 54 Q60 50 46 54 L44 60 Q40 60 40 56 Z",
  asym:     "M30 60 Q30 38 60 36 Q90 38 90 60 L94 100 L82 100 L80 70 L76 56 Q60 52 44 56 L42 70 L38 78 L36 70 L34 64 Z",
  wolf:     "M28 64 Q28 36 60 34 Q92 36 92 64 L96 112 L82 112 L80 84 L76 60 Q60 56 44 60 L40 84 L38 112 L24 112 Z",
  slick:    "M36 60 Q36 36 60 36 Q88 36 88 62 Q90 76 84 84 L80 60 Q60 56 40 60 L36 84 Q30 76 36 60 Z",
  crop:     "M38 58 Q38 36 60 36 Q84 36 84 58 Q84 64 78 64 L76 56 Q60 52 44 56 L42 64 Q38 64 38 58 Z",
  curtain:  "M32 64 Q32 38 60 36 Q88 38 88 64 L88 100 L82 100 L80 76 Q72 60 60 60 Q48 60 40 76 L38 100 L32 100 Z M50 56 Q56 64 60 64 Q64 64 70 56",
  blunt:    "M32 68 Q32 38 60 36 Q88 38 88 68 L88 110 L80 110 L78 60 Q60 56 42 60 L40 110 L32 110 Z",
};

// Light SVG silhouettes used as preview thumbnails. Abstract on purpose —
// these are pattern hints, not realistic renders.
function HairSilhouette({ id, tone = '#2a1f17', skin = '#e8c8a8' }) {
  const Face = () => (
    <>
      <ellipse cx="60" cy="70" rx="20" ry="26" fill={skin} />
      <rect x="52" y="92" width="16" height="14" rx="4" fill={skin} />
    </>
  );
  const d = HAIR_SVG_PATHS[id];
  return (
    <svg viewBox="0 0 120 120" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <Face />
      {d && <path d={d} fill={tone} />}
    </svg>
  );
}

Object.assign(window, { HAIRCUTS, MOODS, HairSilhouette, HAIR_SVG_PATHS });
