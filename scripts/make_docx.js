/* Build paper/contamsens_manuscript.docx from paper/PAPER_FINAL.md.
 *
 * Parses the markdown subset the paper actually uses (H1-H4, paragraphs,
 * tables, ordered/unordered lists, blockquotes, bold/italic/inline-code,
 * autolinks) and emits a Word manuscript: title page block, abstract,
 * numbered sections, four embedded figures with captions, real tables,
 * appendix, references. Regenerate after editing the manuscript:
 *
 *   node scripts/make_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink, BorderStyle,
  WidthType, ShadingType, PageNumber, HeadingLevel, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const SRC = path.join(ROOT, "paper", "PAPER_FINAL.md");
const OUT = path.join(ROOT, "paper", "contamsens_manuscript.docx");
const FIGDIR = path.join(ROOT, "results", "figures");

const BODY = 22;            // 11pt in half-points
const FONT = "Times New Roman";
const CODE_FONT = "Consolas";
const CONTENT_W = 9360;     // US Letter, 1in margins, in DXA
const IMG_W_PX = 600;       // ~6.25in at 96dpi

// figure insertions: marker substring in a paragraph -> [file, caption]
const FIGURES = [
  ["Figure 1 plots the identified set", "f2_identified_set.png",
   "Figure 1: The sharp identified set for the clean score as the assumed " +
   "lift strength Λ varies, at budgets π ∈ {0.05, 0.15, 0.3} " +
   "(synthetic data with known ground truth). The observed score (dashed) is " +
   "the upper endpoint; the true clean score θ (solid) lies inside the " +
   "band at the true (Λ, π), and the set collapses to the observed " +
   "mean as Λ → 0."],
  ["Fig. 2 / f19", "f19_corpusmix.png",
   "Figure 2: Corpus dilution is not one thing. Interleaving each leaked " +
   "chunk with mix ∈ {0, 4, 20} neutral wikitext chunks holds per-item " +
   "exposures fixed while total training grows. A single exposure (dose 1) " +
   "decays into the calibrated field range [0.10, 0.45]; repeated exposures " +
   "(dose 4) consolidate toward complete memorization " +
   "(Λ̂ = 0.98). Session-reproduction noise on " +
   "Λ̂ is ≤ 0.039."],
  ["Fig. 3 / f17", "f17_confirmatory.png",
   "Figure 3: The pre-registered confirmatory audit. Each point is one " +
   "adjacent leaderboard claim's Γ* at π = 0.1; gold bars mark the " +
   "frozen stratum Λ_ref (0.42 HellaSwag, 0.355 elsewhere). Red claims " +
   "fall below the calibrated contamination strength (49/58 = 84.5%); green " +
   "claims survive."],
  ["Fig. 4 / f18", "f18_lambda_sensitivity.png",
   "Figure 4: The headline does not hinge on the frozen constant. Because " +
   "the decision rule depends on (π, Λ) only through B = " +
   "π·Λ, the whole plane collapses to one curve. H4 clears " +
   "its pre-registered 25% bar for every Λ ≥ 0.055, below the " +
   "floor of the calibrated field range (shaded)."],
];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

/* ---------------- inline markdown -> TextRun[] ---------------- */
const STAR = ""; // placeholder for escaped \*
function inline(text, base = {}) {
  text = text.replace(/\\\*/g, STAR);
  const runs = [];
  // tokenize: code spans, bold, italics, autolinks, [text](url)
  const rx = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(<https?:\/\/[^>]+>)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m;
  const plain = (s, extra = {}) => {
    if (!s) return;
    runs.push(new TextRun({ text: s.replace(new RegExp(STAR, "g"), "*"),
                            font: FONT, size: BODY, ...base, ...extra }));
  };
  while ((m = rx.exec(text)) !== null) {
    plain(text.slice(last, m.index));
    const tok = m[0];
    if (m[1]) {
      runs.push(new TextRun({ text: tok.slice(1, -1).replace(new RegExp(STAR, "g"), "*"),
                              font: CODE_FONT, size: BODY - 2, ...base }));
    } else if (m[2]) plain(tok.slice(2, -2), { bold: true });
    else if (m[3]) plain(tok.slice(1, -1), { italics: true });
    else if (m[4]) {
      const url = tok.slice(1, -1);
      runs.push(new ExternalHyperlink({
        link: url,
        children: [new TextRun({ text: url, font: FONT, size: BODY,
                                 color: "0563C1", underline: {}, ...base })],
      }));
    } else if (m[5]) {
      const mm = tok.match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (/^https?:/.test(mm[2])) {
        runs.push(new ExternalHyperlink({
          link: mm[2],
          children: [new TextRun({ text: mm[1], font: FONT, size: BODY,
                                   color: "0563C1", underline: {}, ...base })],
        }));
      } else plain(mm[1]);
    }
    last = m.index + tok.length;
  }
  plain(text.slice(last));
  return runs;
}

/* ---------------- block-level parse ---------------- */
const src = fs.readFileSync(SRC, "utf8").split(/\r?\n/);
const children = [];
let i = 0, sawTitle = false, figQueue = [...FIGURES];

const border = { style: BorderStyle.SINGLE, size: 2, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function para(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 300 },
    ...opts,
    children: inline(text, opts.baseRun || {}),
  });
}

function emitFigure(file, caption) {
  const fp = path.join(FIGDIR, file);
  const { w, h } = pngSize(fp);
  const width = IMG_W_PX, height = Math.round((h / w) * IMG_W_PX);
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(fp),
      transformation: { width, height },
      altText: { title: caption.slice(0, 60), description: caption, name: file },
    })],
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 },
    children: [new TextRun({ text: caption, font: FONT, size: BODY - 3 })],
  }));
}

function table(rows) {
  const cells = rows.map(r =>
    r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const header = cells[0];
  const body = cells.slice(2); // skip |---| separator
  const ncol = header.length;
  const colw = Math.floor(CONTENT_W * 0.7 / ncol);
  const tw = colw * ncol;
  const mkRow = (r, isHead) => new TableRow({
    children: r.map(c => new TableCell({
      borders, width: { size: colw, type: WidthType.DXA },
      shading: isHead ? { fill: "EFEFEF", type: ShadingType.CLEAR } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: inline(c, isHead ? { bold: true } : {}),
      })],
    })),
  });
  return new Table({
    alignment: AlignmentType.CENTER,
    width: { size: tw, type: WidthType.DXA },
    columnWidths: Array(ncol).fill(colw),
    rows: [mkRow(header, true), ...body.map(r => mkRow(r, false))],
  });
}

while (i < src.length) {
  const line = src[i];

  if (/^\s*$/.test(line) || /^---\s*$/.test(line)) { i++; continue; }

  // skip the draft-status italic block; the docx gets a dated title page line
  if (/^\*Draft v/.test(line)) {
    while (i < src.length && !/^\s*$/.test(src[i])) i++;
    continue;
  }

  if (/^# /.test(line)) {
    const text = line.slice(2);
    if (!sawTitle) {
      sawTitle = true;
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 240 },
        children: inline(text, { bold: true, size: 34 }),
      }));
    } else {
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
        children: inline(text),
      }));
    }
    i++; continue;
  }
  if (/^## /.test(line)) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_1, children: inline(line.slice(3)) }));
    i++; continue;
  }
  if (/^### /.test(line)) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_2, children: inline(line.slice(4)) }));
    i++; continue;
  }
  if (/^#### /.test(line)) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_3, children: inline(line.slice(5)) }));
    i++; continue;
  }

  // author / affiliation lines right after the title
  if (/^\*\*Tanvir/.test(line)) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: inline(line, { size: BODY + 2 }) }));
    i++; continue;
  }
  if (/^¹ /.test(line)) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 480 },
      children: inline(line, { size: BODY - 2 }) }));
    i++; continue;
  }

  if (/^\|/.test(line)) {                      // table
    const rows = [];
    while (i < src.length && /^\|/.test(src[i])) rows.push(src[i++]);
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }

  if (/^> /.test(line)) {                      // blockquote (display math)
    const qs = [];
    while (i < src.length && /^> /.test(src[i])) qs.push(src[i++].slice(2));
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
      children: inline(qs.join(" ")),
    }));
    continue;
  }

  if (/^\d+\. /.test(line)) {                  // ordered list item (+wraps)
    let item = line.replace(/^\d+\. /, "");
    i++;
    while (i < src.length && /^ {3}\S/.test(src[i])) item += " " + src[i++].trim();
    children.push(new Paragraph({
      numbering: { reference: "nums", level: 0 },
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 },
      children: inline(item),
    }));
    continue;
  }
  if (/^- /.test(line)) {                      // bullet item (+wraps)
    let item = line.slice(2);
    i++;
    while (i < src.length && /^ {2,}\S/.test(src[i]) && !/^- /.test(src[i]))
      item += " " + src[i++].trim();
    children.push(new Paragraph({
      numbering: { reference: "bullets", level: 0 },
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 },
      children: inline(item),
    }));
    continue;
  }

  // regular paragraph: accumulate until blank/structural line
  const buf = [line];
  i++;
  while (i < src.length && !/^\s*$/.test(src[i]) &&
         !/^(#|\||>|- |\d+\. |---)/.test(src[i])) buf.push(src[i++]);
  const text = buf.join(" ");
  children.push(para(text));

  // queued figure insertion after the referencing paragraph
  for (let q = 0; q < figQueue.length; q++) {
    if (text.includes(figQueue[q][0])) {
      emitFigure(figQueue[q][1], figQueue[q][2]);
      figQueue.splice(q, 1);
      break;
    }
  }
}

if (figQueue.length) {
  console.error("UNPLACED FIGURES:", figQueue.map(f => f[1]));
  process.exit(1);
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        run: { size: 28, bold: true, font: FONT },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        run: { size: 25, bold: true, font: FONT },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        run: { size: BODY, bold: true, italics: true, font: FONT },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET,
        text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL,
        text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
    } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Contamination Sensitivity Analysis for Benchmark Claims",
                               font: FONT, size: 18, italics: true })],
    })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })],
    })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${path.relative(ROOT, OUT)} (${(buf.length / 1024).toFixed(0)} KB)`);
});
