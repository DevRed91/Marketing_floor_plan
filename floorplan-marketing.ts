/**
 * Floor plan -> marketing floor plan (trial), TypeScript version.
 *
 * Same pipeline and same extraction.json format as the Python version, so the two are interchangeable:
 *   1. Input check   - resolution, and Gemini classifies the input (CAD plan / photo / sketch / multi-unit ...).
 *   2. Extraction    - Gemini returns the plan as JSON (rooms, sizes, doors, windows, open-plan links,
 *                      exterior zones, furniture, ambiguous symbols, boxes around every text label).
 *   3. Confirmation  - review the extraction (edit extraction.json and rerun with --extraction if needed).
 *   4. Cleaning      - code crops away dimension chains and erases labels/tags/compass, keeping walls intact.
 *   5. Generation    - N candidates from the Gemini image model with a locked prompt (no text, no people).
 *   6. Verification  - Gemini compares each candidate with the source; layout errors fail the candidate.
 *   7. Overlay       - room names, sizes, compass, disclaimer and optional logo are drawn in code (SVG).
 *   8. Review output - side-by-side comparison + report.json. If nothing passes, no final image is produced.
 *
 * Usage:
 *   export GEMINI_API_KEY=...
 *   npx tsx floorplan-marketing.ts plan.png
 *   npx tsx floorplan-marketing.ts plan.png --candidates 4 --logo logo.png --yes
 *   npx tsx floorplan-marketing.ts plan.png --extraction out/plan/extraction.json
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import readline from "node:readline/promises";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { GoogleGenAI, Modality, type Part } from "@google/genai";
import sharp, { type OverlayOptions } from "sharp";
import { z } from "zod";

// --------------------------------------------------------------------------------------
// Configuration — pin exact model versions; rerun your test set before changing them.
// --------------------------------------------------------------------------------------
export const TEXT_MODEL = process.env.FP_TEXT_MODEL ?? "gemini-3.5-flash";
export const IMAGE_MODEL = process.env.FP_IMAGE_MODEL ?? "gemini-3.1-flash-image";

const MIN_SIDE_PX = 800;
const SUPPORTED_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
const REJECTED_INPUT_TYPES = new Set(["photo_of_paper", "hand_sketch", "multi_unit_sheet", "not_a_floor_plan"]);
const DISCLAIMER = "Illustrative only. Furniture, finishes and decor are not included. Dimensions are approximate.";
const FONT_FAMILY = "DejaVu Sans, Arial, Helvetica, sans-serif";
const INK = "#2d2d2d";

// One house style. Change it here, not per request.
const HOUSE_STYLE = `\
- Floors: light natural oak (herringbone) in living areas, hall and bedrooms; light stone tiles in bathrooms;
  light stone or tile in the kitchen if it is a separate room; outdoor wood decking on balconies and terraces.
- Wall tops: solid dark charcoal, crisp and uniform, so the layout reads clearly.
- Palette: warm neutrals — cream, beige, oak, soft greys, a few muted green plants.
- Lighting: soft, even daylight from above with gentle, short contact shadows. No dramatic lighting.
- Finish: clean, premium real-estate marketing look, photorealistic but uncluttered.`;

// --------------------------------------------------------------------------------------
// Structured-output schemas (snake_case so extraction.json matches the Python version)
// --------------------------------------------------------------------------------------
const Box = z.array(z.number().int());

const Room = z.object({
  name: z.string().describe("Room name as written on the plan, or a sensible name if unlabelled."),
  size_label: z.string().describe("Size exactly as written on the plan, e.g. '5.80 × 4.80'. Empty string if none."),
  label_box: Box.describe(
    "[ymin, xmin, ymax, xmax] normalised 0-1000: box around this room's name and size text. " +
      "If the room has no text, a small box at the visual centre of the room.",
  ),
  room_box: Box.describe("[ymin, xmin, ymax, xmax] normalised 0-1000: the room's interior, wall to wall."),
  open_to: z.array(z.string()).describe("Names of other rooms this room connects to with NO wall between them."),
});

const TextItem = z.object({
  text: z.string(),
  box: Box.describe("[ymin, xmin, ymax, xmax] normalised 0-1000, tight around the text."),
});

const AmbiguousItem = z.object({
  description: z.string().describe("What the symbol looks like, e.g. '7 circles in a row'."),
  location: z.string().describe("Where it is, e.g. 'along the balcony'."),
  likely_meaning: z.string().describe("Best interpretation, e.g. 'round planters with small trees'."),
});

export const Extraction = z.object({
  input_type: z.enum(["cad_drawing", "clean_scan", "photo_of_paper", "hand_sketch", "multi_unit_sheet", "not_a_floor_plan"]),
  single_unit: z.boolean(),
  rooms: z.array(Room),
  door_count: z.number().int(),
  window_count: z.number().int(),
  entrance: z.string().describe("Where the main entrance door is, e.g. 'west wall of the living room'."),
  exterior_zones: z
    .array(z.string())
    .describe("Areas inside the plan's outer rectangle that are NOT part of the unit (outside space, voids), by position and size."),
  furniture: z.array(z.string()).describe("Each drawn furniture/fixture item with its room and position."),
  ambiguous_items: z.array(AmbiguousItem),
  text_items: z
    .array(TextItem)
    .describe(
      "EVERY piece of text or annotation drawn on or inside the plan outline: room names, sizes, door/window tags " +
        "(e.g. D1, W2) including their circles, scale notes. Exclude dimension chains outside the walls.",
    ),
  compass_box: Box.describe("[ymin, xmin, ymax, xmax] 0-1000 around the north arrow, or [] if none."),
  north_direction: z.enum(["up", "down", "left", "right", "unknown"]),
  notes: z.array(z.string()).describe("Inconsistencies, e.g. dimension chains that disagree. Empty if none."),
});
export type Extraction = z.infer<typeof Extraction>;

const Judgement = z.object({
  missing_rooms: z.array(z.string()).describe("Source rooms not clearly present in the render."),
  extra_rooms: z.array(z.string()).describe("Enclosed spaces in the render that do not exist in the source."),
  walls_added: z.array(z.string()),
  walls_removed_or_moved: z.array(z.string()),
  doors_missing_or_moved: z.array(z.string()),
  windows_missing_or_moved: z.array(z.string()),
  door_count: z.number().int(),
  window_count: z.number().int(),
  open_plan_preserved: z.boolean(),
  exterior_filled_as_interior: z.boolean(),
  text_or_numbers_present: z.boolean(),
  people_present: z.boolean(),
  top_down_orthographic: z.boolean(),
  fidelity_score: z.number().int().describe("0-10, where 10 means the layout is identical to the source."),
});
type Judgement = z.infer<typeof Judgement>;

/** Zod -> plain JSON Schema for Gemini (drop keywords Gemini does not need). */
function toGeminiSchema(schema: z.ZodType): unknown {
  const strip = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(strip);
    if (node && typeof node === "object") {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(node)) {
        if (k === "$schema" || k === "additionalProperties") continue;
        out[k] = strip(v);
      }
      return out;
    }
    return node;
  };
  return strip(z.toJSONSchema(schema));
}

// --------------------------------------------------------------------------------------
// Gemini calls
// --------------------------------------------------------------------------------------
function makeClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) fail("Set GEMINI_API_KEY (or GOOGLE_API_KEY) first.");
  return new GoogleGenAI({ apiKey });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function withRetry<T>(fn: () => Promise<T>, label: string, attempts = 3): Promise<T> {
  for (let i = 0; ; i++) {
    try {
      return await fn();
    } catch (e) {
      if (i >= attempts - 1) throw e;
      const wait = 3 * 2 ** i;
      console.log(`  ! ${label} failed (${(e as Error).message}); retrying in ${wait}s`);
      await sleep(wait * 1000);
    }
  }
}

const imagePart = (png: Buffer): Part => ({ inlineData: { mimeType: "image/png", data: png.toString("base64") } });

async function structuredCall<S extends z.ZodType>(ai: GoogleGenAI, parts: Part[], schema: S): Promise<z.infer<S>> {
  const resp = await ai.models.generateContent({
    model: TEXT_MODEL,
    contents: [{ role: "user", parts }],
    config: { responseMimeType: "application/json", responseJsonSchema: toGeminiSchema(schema) },
  });
  return schema.parse(JSON.parse(resp.text ?? ""));
}

const EXTRACTION_PROMPT = `\
You are reading an architectural floor plan for a real-estate marketing pipeline. Be precise and literal.
- Classify the input type honestly. A clean computer-drawn plan is 'cad_drawing'; a scanned clean plan is 'clean_scan';
  a camera photo of paper (perspective, shadows) is 'photo_of_paper'.
- single_unit is false if the image shows several apartments or several floors.
- List every room, including balconies, halls, corridors and storage. Copy names and sizes exactly as written.
- open_to: only list connections with no wall at all between rooms (open-plan). A door is not open-plan.
- Count every door (including the entrance and balcony doors) and every window.
- exterior_zones: parts of the drawing's overall rectangle that are outside the unit (e.g. an empty corner).
- ambiguous_items: any symbol whose meaning is unclear (circles, long thin objects, hatched areas).
- text_items: every text label and tag inside or on the plan outline, with tight boxes, so they can be erased.
All boxes use [ymin, xmin, ymax, xmax] normalised to 0-1000 over the whole image.`;

export function buildGenerationPrompt(ex: Extraction): string {
  const list = (items: string[], empty: string) => (items.length ? items.map((s) => `  - ${s}`).join("\n") : `  - ${empty}`);
  const rooms = list(ex.rooms.map((r) => `${r.name} (${r.size_label || "size not labelled"})`), "none");
  const openPlan = list(
    ex.rooms.filter((r) => r.open_to.length).map((r) => `${r.name} is open to ${r.open_to.join(", ")}; there is NO wall between them, keep it open`),
    "none",
  );
  const exterior = list(ex.exterior_zones, "none");
  const furniture = list(ex.furniture, "furnish sensibly for each room's purpose");
  const ambiguous = list(ex.ambiguous_items.map((a) => `${a.description} (${a.location}): render as ${a.likely_meaning}`), "none");
  return `\
Restyle the attached architectural floor plan into a premium marketing floor plan render of THIS EXACT plan.
Think of it as painting materials and furniture onto the drawing, not designing a new apartment.

GEOMETRY — MUST NOT CHANGE
- Keep every wall exactly where it is, with the same length and thickness. Do not add, remove, extend, shorten or move any wall.
- Keep every door opening and every window in the same position and size. Doors: ${ex.door_count}. Windows: ${ex.window_count}.
- Keep every room's shape, size and position identical. Rooms:
${rooms}
- Open-plan connections:
${openPlan}
- Areas that are OUTSIDE the unit — render as plain light-grey ground with no flooring, walls or furniture:
${exterior}
- Camera: straight top-down orthographic view, no tilt, no perspective. Keep the same framing and position as the input image.

FURNISHING
- Follow the furniture and fixtures already drawn, in the same positions and orientations:
${furniture}
- Interpret these symbols as:
${ambiguous}

STYLE
${HOUSE_STYLE}

DO NOT INCLUDE
- Any text, letters, numbers, labels, dimension lines, tags, symbols, compass, logos or watermarks.
- Any people or animals.
- Anything beyond the unit's outline.`;
}

async function generateCandidate(ai: GoogleGenAI, cleanPng: Buffer, prompt: string, aspectRatio: string, size: string): Promise<Buffer | null> {
  const resp = await ai.models.generateContent({
    model: IMAGE_MODEL,
    contents: [{ role: "user", parts: [imagePart(cleanPng), { text: prompt }] }],
    config: { responseModalities: [Modality.IMAGE], imageConfig: { aspectRatio, imageSize: size } },
  });
  for (const cand of resp.candidates ?? []) {
    for (const part of cand.content?.parts ?? []) {
      if (part.inlineData?.data) return sharp(Buffer.from(part.inlineData.data, "base64")).png().toBuffer();
    }
  }
  return null;
}

async function judge(ai: GoogleGenAI, sourcePng: Buffer, candidatePng: Buffer, ex: Extraction): Promise<Judgement> {
  const rooms = ex.rooms.map((r) => r.name).join(", ");
  const openPlan = ex.rooms.filter((r) => r.open_to.length).map((r) => `${r.name} open to ${r.open_to.join(", ")}`).join("; ") || "none";
  const prompt = `\
You are a strict QA reviewer. Image 1 is the SOURCE floor plan (technical drawing). Image 2 is a CANDIDATE
marketing render generated from it. Decide whether the render faithfully represents the source LAYOUT.
Ignore colours, materials, lighting and furniture design. Check only walls, rooms, doors, windows and framing.

Source facts: rooms = ${rooms}. Doors = ${ex.door_count}. Windows = ${ex.window_count}.
Open-plan = ${openPlan}. Exterior zones = ${ex.exterior_zones.join("; ") || "none"}.

Report every wall that was added, removed or moved; every missing or moved door and window; missing or extra rooms;
whether open-plan areas stayed open; whether an exterior zone was filled in as interior; whether any text or numbers
appear; whether people appear; whether the view is straight top-down. Count doors and windows visible in the render.
fidelity_score: 10 = identical layout, 7 = small visual drift but every wall/door/window in place, below 5 = wrong layout.`;
  return structuredCall(
    ai,
    [{ text: "SOURCE PLAN:" }, imagePart(sourcePng), { text: "CANDIDATE RENDER:" }, imagePart(candidatePng), { text: prompt }],
    Judgement,
  );
}

function judgementProblems(j: Judgement, ex: Extraction, tolerance: number): string[] {
  const problems: string[] = [];
  const lists: [string, string[]][] = [
    ["missing rooms", j.missing_rooms],
    ["extra rooms", j.extra_rooms],
    ["walls added", j.walls_added],
    ["walls removed/moved", j.walls_removed_or_moved],
    ["doors missing/moved", j.doors_missing_or_moved],
    ["windows missing/moved", j.windows_missing_or_moved],
  ];
  for (const [label, items] of lists) if (items.length) problems.push(`${label}: ${items.join("; ")}`);
  if (Math.abs(j.door_count - ex.door_count) > tolerance) problems.push(`door count ${j.door_count} vs source ${ex.door_count}`);
  if (Math.abs(j.window_count - ex.window_count) > tolerance) problems.push(`window count ${j.window_count} vs source ${ex.window_count}`);
  if (ex.rooms.some((r) => r.open_to.length) && !j.open_plan_preserved) problems.push("open-plan area was closed off");
  if (j.exterior_filled_as_interior) problems.push("exterior zone rendered as interior");
  if (j.people_present) problems.push("people present");
  if (!j.top_down_orthographic) problems.push("not a straight top-down view");
  if (j.text_or_numbers_present) problems.push("text or numbers present in the render");
  return problems;
}

// --------------------------------------------------------------------------------------
// Deterministic image processing (raw RGB buffers + sharp)
// --------------------------------------------------------------------------------------
export interface RGB {
  data: Buffer; // 3 channels, row-major
  width: number;
  height: number;
}
type Rect = [number, number, number, number]; // x0, y0, x1, y1 (pixels, exclusive end)

export async function loadRGB(input: string | Buffer): Promise<RGB> {
  const { data, info } = await sharp(input).flatten({ background: "#ffffff" }).toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new Error(`expected 3 channels, got ${info.channels}`);
  return { data, width: info.width, height: info.height };
}

const rawSharp = (img: RGB) => sharp(img.data, { raw: { width: img.width, height: img.height, channels: 3 } });

export function boxToPx(box: number[], w: number, h: number): Rect | null {
  if (box.length !== 4) return null;
  const [ymin, xmin, ymax, xmax] = box;
  const [x0, x1] = [Math.floor((xmin / 1000) * w), Math.floor((xmax / 1000) * w)].sort((a, b) => a - b);
  const [y0, y1] = [Math.floor((ymin / 1000) * h), Math.floor((ymax / 1000) * h)].sort((a, b) => a - b);
  return [Math.max(0, x0), Math.max(0, y0), Math.min(w, x1), Math.min(h, y1)];
}

/** Separable binary erosion/dilation with a k-wide window along rows or columns. */
function morph1D(mask: Uint8Array, w: number, h: number, k: number, horizontal: boolean, mode: "erode" | "dilate"): Uint8Array {
  const out = new Uint8Array(w * h);
  const a = Math.floor(k / 2);
  const lo = mode === "erode" ? -a : -(k - 1 - a);
  const hi = mode === "erode" ? k - 1 - a : a;
  const n = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const pref = new Int32Array(n + 1);
  for (let l = 0; l < lines; l++) {
    for (let i = 0; i < n; i++) pref[i + 1] = pref[i] + mask[horizontal ? l * w + i : i * w + l];
    for (let i = 0; i < n; i++) {
      const s = Math.max(0, i + lo);
      const e = Math.min(n - 1, i + hi);
      const sum = pref[e + 1] - pref[s];
      out[horizontal ? l * w + i : i * w + l] = mode === "erode" ? (sum === e - s + 1 ? 1 : 0) : sum > 0 ? 1 : 0;
    }
  }
  return out;
}

function openMask(mask: Uint8Array, w: number, h: number, k: number): Uint8Array {
  let m = morph1D(mask, w, h, k, true, "erode");
  m = morph1D(m, w, h, k, false, "erode");
  m = morph1D(m, w, h, k, true, "dilate");
  return morph1D(m, w, h, k, false, "dilate");
}

/** Stroke thickness per dark pixel = min(horizontal run, vertical run); walls are the thickest strokes. */
function estimateWallThickness(dark: Uint8Array, w: number, h: number): number {
  const hRun = new Uint16Array(w * h);
  for (let y = 0; y < h; y++) {
    let x = 0;
    while (x < w) {
      if (!dark[y * w + x]) { x++; continue; }
      let e = x;
      while (e < w && dark[y * w + e]) e++;
      for (let i = x; i < e; i++) hRun[y * w + i] = Math.min(e - x, 65535);
      x = e;
    }
  }
  const hist = new Uint32Array(1001);
  let total = 0;
  for (let x = 0; x < w; x++) {
    let y = 0;
    while (y < h) {
      if (!dark[y * w + x]) { y++; continue; }
      let e = y;
      while (e < h && dark[e * w + x]) e++;
      for (let i = y; i < e; i++) {
        hist[Math.min(1000, Math.min(e - y, hRun[i * w + x]))]++;
        total++;
      }
      y = e;
    }
  }
  if (!total) throw new Error("no dark lines found");
  let acc = 0;
  for (let t = 0; t <= 1000; t++) {
    acc += hist[t];
    if (acc >= 0.9 * total) return t;
  }
  return 1000;
}

export function detectWalls(img: RGB): { bbox: Rect; walls: Uint8Array; k: number } {
  const { width: w, height: h, data } = img;
  const dark = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const g = 0.299 * data[3 * i] + 0.587 * data[3 * i + 1] + 0.114 * data[3 * i + 2];
    dark[i] = g < 110 ? 1 : 0;
  }
  const k = Math.max(3, Math.round(estimateWallThickness(dark, w, h) * 0.6));
  const opened = openMask(dark, w, h, k);

  // 8-connected components; keep the big ones (walls), drop arrowheads and specks.
  const labels = new Int32Array(w * h);
  const comps: { area: number; x0: number; y0: number; x1: number; y1: number }[] = [];
  const stack: number[] = [];
  for (let start = 0; start < w * h; start++) {
    if (!opened[start] || labels[start]) continue;
    const id = comps.length + 1;
    const c = { area: 0, x0: w, y0: h, x1: 0, y1: 0 };
    labels[start] = id;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop()!;
      const px = p % w, py = (p - px) / w;
      c.area++;
      c.x0 = Math.min(c.x0, px); c.y0 = Math.min(c.y0, py);
      c.x1 = Math.max(c.x1, px + 1); c.y1 = Math.max(c.y1, py + 1);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (opened[q] && !labels[q]) { labels[q] = id; stack.push(q); }
      }
    }
    comps.push(c);
  }
  if (!comps.length) throw new Error("no wall components found");
  const maxArea = Math.max(...comps.map((c) => c.area));
  const keep = new Set<number>();
  comps.forEach((c, i) => { if (c.area >= 0.01 * maxArea) keep.add(i + 1); });
  const kept = comps.filter((_, i) => keep.has(i + 1));
  const bbox: Rect = [Math.min(...kept.map((c) => c.x0)), Math.min(...kept.map((c) => c.y0)), Math.max(...kept.map((c) => c.x1)), Math.max(...kept.map((c) => c.y1))];
  const walls = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) walls[i] = keep.has(labels[i]) ? 1 : 0;
  return { bbox, walls, k };
}

/** Erase labels/tags/compass (restoring walls) and crop away dimension chains. */
export async function cleanPlan(img: RGB, ex: Extraction): Promise<{ clean: RGB; crop: Rect }> {
  const { width: w, height: h } = img;
  let bbox: Rect = [0, 0, w, h];
  let walls: Uint8Array = new Uint8Array(w * h);
  let k = 4;
  try {
    ({ bbox, walls, k } = detectWalls(img));
  } catch {
    /* fall back to the whole image */
  }
  const out = Buffer.from(img.data);
  const paint = (x0: number, y0: number, x1: number, y1: number) => {
    for (let y = Math.max(0, y0); y < Math.min(h, y1); y++) out.fill(255, 3 * (y * w + Math.max(0, x0)), 3 * (y * w + Math.min(w, x1)));
  };
  const pad = 4;
  const boxes = [...ex.text_items.map((t) => t.box), ...(ex.compass_box.length ? [ex.compass_box] : [])];
  for (const b of boxes) {
    const r = boxToPx(b, w, h);
    if (r) paint(r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad);
  }
  const [bx0, by0, bx1, by1] = bbox;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x < bx0 || x >= bx1 || y < by0 || y >= by1) out.fill(255, 3 * i, 3 * i + 3); // outside the wall outline
      else if (walls[i]) img.data.copy(out, 3 * i, 3 * i, 3 * i + 3); // never erase a wall
    }
  }
  const margin = 2 * k;
  const crop: Rect = [Math.max(0, bx0 - margin), Math.max(0, by0 - margin), Math.min(w, bx1 + margin), Math.min(h, by1 + margin)];
  const { data, info } = await rawSharp({ data: out, width: w, height: h })
    .extract({ left: crop[0], top: crop[1], width: crop[2] - crop[0], height: crop[3] - crop[1] })
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { clean: { data, width: info.width, height: info.height }, crop };
}

export function closestRatio(w: number, h: number): string {
  const val = (s: string) => { const [a, b] = s.split(":").map(Number); return a / b; };
  return SUPPORTED_RATIOS.reduce((best, s) => (Math.abs(Math.log(w / h) - Math.log(val(s))) < Math.abs(Math.log(w / h) - Math.log(val(best))) ? s : best));
}

export async function padToRatio(img: RGB, ratio: string): Promise<{ png: Buffer; size: [number, number]; offset: [number, number] }> {
  const [a, b] = ratio.split(":").map(Number);
  const target = a / b;
  const [w, h] = [img.width, img.height];
  const [nw, nh] = w / h < target ? [Math.round(h * target), h] : [w, Math.round(w / target)];
  const left = Math.floor((nw - w) / 2), top = Math.floor((nh - h) / 2);
  const png = await rawSharp(img)
    .extend({ left, top, right: nw - w - left, bottom: nh - h - top, background: "#ffffff" })
    .png()
    .toBuffer();
  return { png, size: [nw, nh], offset: [left, top] };
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
// No text measurement in SVG, so estimate widths (average glyph width as a fraction of font size).
const textWidth = (s: string, px: number, bold: boolean) => s.length * px * (bold ? 0.72 : 0.58);

function labelSvg(cx: number, cy: number, name: string, sizeLabel: string, namePx: number, sizePx: number, room: Rect | null): string {
  const layout = (np: number, sp: number) => {
    const lines = [{ text: name.toUpperCase(), px: np, bold: true }, ...(sizeLabel ? [{ text: sizeLabel, px: sp, bold: false }] : [])];
    const gap = sp * 0.35, pad = sp * 0.55;
    const tw = Math.max(...lines.map((l) => textWidth(l.text, l.px, l.bold)));
    const th = lines.reduce((s, l) => s + l.px, 0) + gap * (lines.length - 1);
    return { lines, gap, pad, bw: tw + 2 * pad, bh: th + 2 * pad };
  };
  let L = layout(namePx, sizePx);
  if (room) {
    const [rx0, ry0, rx1, ry1] = room;
    while ((L.bw > 0.92 * (rx1 - rx0) || L.bh > 0.92 * (ry1 - ry0)) && sizePx > 8) {
      namePx = Math.floor(namePx * 0.9); sizePx = Math.floor(sizePx * 0.9);
      L = layout(namePx, sizePx);
    }
    if (rx1 - rx0 > L.bw) cx = Math.min(Math.max(cx, rx0 + L.bw / 2), rx1 - L.bw / 2);
    if (ry1 - ry0 > L.bh) cy = Math.min(Math.max(cy, ry0 + L.bh / 2), ry1 - L.bh / 2);
  }
  let svg = `<rect x="${cx - L.bw / 2}" y="${cy - L.bh / 2}" width="${L.bw}" height="${L.bh}" rx="${L.pad}" fill="#ffffff" fill-opacity="0.8"/>`;
  let y = cy - L.bh / 2 + L.pad;
  for (const l of L.lines) {
    svg += `<text x="${cx}" y="${y + l.px * 0.82}" font-family="${FONT_FAMILY}" font-size="${l.px}" font-weight="${l.bold ? 700 : 400}" fill="${INK}" text-anchor="middle">${esc(l.text)}</text>`;
    y += l.px + L.gap;
  }
  return svg;
}

function compassSvg(cx: number, cy: number, r: number, direction: string): string {
  const angle = { up: 0, right: 90, down: 180, left: 270 }[direction] ?? 0;
  const rad = (angle * Math.PI) / 180;
  const nx = cx + r * 1.45 * Math.sin(rad), ny = cy - r * 1.45 * Math.cos(rad);
  const pts = [[0, -r * 0.72], [-r * 0.32, r * 0.45], [0, r * 0.2], [r * 0.32, r * 0.45]].map(([x, y]) => `${cx + x},${cy + y}`).join(" ");
  return (
    `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#ffffff" fill-opacity="0.86" stroke="${INK}" stroke-width="${Math.max(2, r / 12)}"/>` +
    `<polygon points="${pts}" fill="${INK}" transform="rotate(${angle} ${cx} ${cy})"/>` +
    `<text x="${nx}" y="${ny + r * 0.22}" font-family="${FONT_FAMILY}" font-size="${r * 0.6}" font-weight="700" fill="${INK}" text-anchor="middle">N</text>`
  );
}

/** Cut the padding back off the render, then draw labels/compass/footer in code. */
export async function composeFinal(
  renderPng: Buffer, paddedSize: [number, number], padOff: [number, number], crop: Rect,
  origSize: [number, number], ex: Extraction, logo?: string,
): Promise<Buffer> {
  const meta = await sharp(renderPng).metadata();
  const [rw, rh] = [meta.width!, meta.height!];
  const sx = rw / paddedSize[0], sy = rh / paddedSize[1];
  const [cx0, cy0, cx1, cy1] = crop;
  const left = Math.round(padOff[0] * sx), top = Math.round(padOff[1] * sy);
  const W = Math.min(Math.round((cx1 - cx0) * sx), rw - left);
  const H = Math.min(Math.round((cy1 - cy0) * sy), rh - top);
  const [ow, oh] = origSize;
  const toPlan = (px: number, py: number): [number, number] => [(px - cx0) * sx, (py - cy0) * sy];

  let svg = "";
  const namePx = Math.max(12, Math.floor(W * 0.016)), sizePx = Math.max(10, Math.floor(W * 0.012));
  for (const room of ex.rooms) {
    const b = boxToPx(room.label_box, ow, oh);
    if (!b) continue;
    const [x, y] = toPlan((b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
    const rb = boxToPx(room.room_box, ow, oh);
    const roomRect: Rect | null = rb && rb[2] > rb[0] && rb[3] > rb[1] ? [...toPlan(rb[0], rb[1]), ...toPlan(rb[2], rb[3])] : null;
    if (x >= 0 && x <= W && y >= 0 && y <= H) svg += labelSvg(x, y, room.name, room.size_label, namePx, sizePx, roomRect);
  }
  const r = Math.max(14, Math.floor(W * 0.028));
  const cb = ex.compass_box.length ? boxToPx(ex.compass_box, ow, oh) : null;
  let [x, y] = cb ? toPlan((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2) : [W - 2.2 * r, 2.4 * r];
  x = Math.min(Math.max(x, 1.6 * r), W - 1.6 * r);
  y = Math.min(Math.max(y, 2.2 * r), H - 1.6 * r);
  svg += compassSvg(x, y, r, ex.north_direction);

  const footerH = Math.max(40, Math.floor(W * 0.055));
  const footerPx = Math.max(10, Math.floor(footerH * 0.28));
  const overlay = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H + footerH}">${svg}` +
      `<text x="${footerH * 0.4}" y="${H + footerH / 2 + footerPx * 0.35}" font-family="${FONT_FAMILY}" font-size="${footerPx}" fill="#5f5f5f">${esc(DISCLAIMER)}</text></svg>`,
  );
  const layers: OverlayOptions[] = [{ input: overlay, left: 0, top: 0 }];
  if (logo) {
    const lg = await sharp(logo).resize({ width: Math.floor(W * 0.22), height: Math.floor(footerH * 0.7), fit: "inside" }).png().toBuffer();
    const lm = await sharp(lg).metadata();
    layers.push({ input: lg, left: W - lm.width! - Math.floor(footerH * 0.4), top: H + Math.floor((footerH - lm.height!) / 2) });
  }
  return sharp(renderPng)
    .extract({ left, top, width: W, height: H })
    .flatten({ background: "#ffffff" })
    .extend({ bottom: footerH, background: "#ffffff" })
    .composite(layers)
    .png()
    .toBuffer();
}

export async function sideBySide(leftPng: Buffer, rightPng: Buffer): Promise<Buffer> {
  const rm = await sharp(rightPng).metadata();
  const h = rm.height!;
  const left = await sharp(leftPng).resize({ height: h }).png().toBuffer();
  const lw = (await sharp(left).metadata()).width!;
  const gap = Math.floor(h * 0.02);
  return sharp({ create: { width: lw + gap + rm.width!, height: h, channels: 3, background: "#ffffff" } })
    .composite([{ input: left, left: 0, top: 0 }, { input: rightPng, left: lw + gap, top: 0 }])
    .png()
    .toBuffer();
}

// --------------------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------------------
function fail(msg: string, code = 1): never {
  console.error(msg);
  process.exit(code);
}

function printExtraction(ex: Extraction) {
  console.log(`\n  Input type: ${ex.input_type}   single unit: ${ex.single_unit}`);
  console.log(`  Doors: ${ex.door_count}   Windows: ${ex.window_count}   Entrance: ${ex.entrance}`);
  console.log("  Rooms:");
  for (const r of ex.rooms) console.log(`    - ${r.name.padEnd(18)} ${r.size_label}${r.open_to.length ? `  (open to ${r.open_to.join(", ")})` : ""}`);
  if (ex.exterior_zones.length) console.log("  Exterior zones:", ex.exterior_zones.join("; "));
  if (ex.ambiguous_items.length) {
    console.log("  Ambiguous symbols (edit 'likely_meaning' in extraction.json to change):");
    for (const a of ex.ambiguous_items) console.log(`    - ${a.description} (${a.location}) -> ${a.likely_meaning}`);
  }
  for (const n of ex.notes) console.log(`  Note: ${n}`);
}

async function main() {
  const { values: o, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: "string" },
      candidates: { type: "string", default: "3" },
      size: { type: "string", default: "2K" },
      logo: { type: "string" },
      extraction: { type: "string" },
      tolerance: { type: "string", default: "0" },
      yes: { type: "boolean", default: false },
      force: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });
  if (o.help || positionals.length !== 1) {
    console.log(`Usage: npx tsx floorplan-marketing.ts <plan.png> [options]
  --out <dir>          Output folder (default ./out/<plan name>)
  --candidates <n>     Number of render candidates (default 3)
  --size 1K|2K|4K      Render resolution (default 2K)
  --logo <file>        Optional logo for the footer
  --extraction <file>  Reuse an existing (possibly hand-edited) extraction.json
  --tolerance <n>      Allowed door/window count difference (default 0)
  --yes                Skip the confirmation prompt
  --force              Bypass input checks and compose the best candidate even if it fails`);
    process.exit(o.help ? 0 : 1);
  }
  if (!["1K", "2K", "4K"].includes(o.size!)) fail("--size must be 1K, 2K or 4K");
  const nCandidates = Number(o.candidates), tolerance = Number(o.tolerance);

  const t0 = Date.now();
  const planPath = positionals[0];
  const out = o.out ?? path.join("out", path.parse(planPath).name);
  await mkdir(path.join(out, "candidates"), { recursive: true });
  const img = await loadRGB(planPath);
  const imgPng = await rawSharp(img).png().toBuffer();
  const ai = makeClient();

  // 1. Input check (resolution)
  if (Math.min(img.width, img.height) < MIN_SIDE_PX && !o.force)
    fail(`Rejected: image is ${img.width}x${img.height}px; need at least ${MIN_SIDE_PX}px on the short side.`);

  // 2. Extraction
  let ex: Extraction;
  if (o.extraction) {
    ex = Extraction.parse(JSON.parse(await readFile(o.extraction, "utf8")));
    console.log(`Loaded extraction from ${o.extraction}`);
  } else {
    console.log(`Reading the plan with ${TEXT_MODEL} ...`);
    ex = await withRetry(() => structuredCall(ai, [imagePart(imgPng), { text: EXTRACTION_PROMPT }], Extraction), "extraction");
  }
  await writeFile(path.join(out, "extraction.json"), JSON.stringify(ex, null, 2));

  // 1b. Input check (content)
  if (!o.force) {
    if (REJECTED_INPUT_TYPES.has(ex.input_type)) fail(`Rejected: input looks like '${ex.input_type}'. Use a clean, computer-drawn plan of one unit.`);
    if (!ex.single_unit) fail("Rejected: the image appears to contain more than one unit or floor. Upload one unit per image.");
    if (!ex.rooms.length) fail("Rejected: no rooms could be read from this plan.");
  }

  // 3. Confirmation
  printExtraction(ex);
  if (!o.yes) {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ans = (await rl.question(`\nContinue with this reading? [Y/n] (to correct it, edit ${path.join(out, "extraction.json")} and rerun with --extraction) `)).trim().toLowerCase();
    rl.close();
    if (!["", "y", "yes"].includes(ans)) fail("Stopped. Edit extraction.json and rerun with --extraction.");
  }

  // 4. Cleaning
  const { clean, crop } = await cleanPlan(img, ex);
  const ratio = closestRatio(clean.width, clean.height);
  const padded = await padToRatio(clean, ratio);
  await writeFile(path.join(out, "clean_plan.png"), await rawSharp(clean).png().toBuffer());
  await writeFile(path.join(out, "clean_plan_padded.png"), padded.png);
  const sourceCrop = await sharp(imgPng).extract({ left: crop[0], top: crop[1], width: crop[2] - crop[0], height: crop[3] - crop[1] }).png().toBuffer();

  // 5. Generation
  const prompt = buildGenerationPrompt(ex);
  await writeFile(path.join(out, "prompt.txt"), prompt);
  console.log(`\nGenerating ${nCandidates} candidates with ${IMAGE_MODEL} (aspect ${ratio}, ${o.size}) ...`);
  const renders = new Map<number, Buffer>();
  await Promise.all(
    Array.from({ length: nCandidates }, (_, k) => k + 1).map(async (i) => {
      const r = await withRetry(() => generateCandidate(ai, padded.png, prompt, ratio, o.size!), `candidate ${i}`);
      if (!r) return console.log(`  candidate ${i}: no image returned (possibly blocked)`);
      await writeFile(path.join(out, "candidates", `candidate_${i}.png`), r);
      renders.set(i, r);
    }),
  );
  if (!renders.size) fail("No candidates were produced. Check the prompt or try again.");

  // 6. Verification
  console.log("Checking each candidate against the source plan ...");
  const results = await Promise.all(
    [...renders.entries()].sort(([a], [b]) => a - b).map(async ([i, r]) => {
      const j = await withRetry(() => judge(ai, padded.png, r, ex), `check ${i}`);
      const problems = judgementProblems(j, ex, tolerance);
      return { candidate: i, passed: !problems.length, score: j.fidelity_score, problems, judgement: j };
    }),
  );
  for (const r of results)
    console.log(`  candidate ${r.candidate}: ${r.passed ? "PASS" : "FAIL"}  score ${r.score}/10${r.passed ? "" : `  -> ${r.problems.join(" | ")}`}`);

  const passing = results.filter((r) => r.passed).sort((a, b) => b.score - a.score);
  const best = passing[0] ?? results.reduce((a, b) => (b.score > a.score ? b : a));
  const report = {
    plan: planPath, text_model: TEXT_MODEL, image_model: IMAGE_MODEL, aspect_ratio: ratio, size: o.size,
    seconds: (Date.now() - t0) / 1000, chosen_candidate: passing.length || o.force ? best.candidate : null, results,
  };

  if (!passing.length && !o.force) {
    await writeFile(path.join(out, "report.json"), JSON.stringify(report, null, 2));
    console.log("\nCouldn't convert this plan reliably: every candidate changed the layout.");
    console.log(`Candidates and reasons are in ${out}. Try again, raise --candidates, or rerun with --force to inspect.`);
    process.exit(2);
  }

  // 7. Overlay + 8. Review output
  const final = await composeFinal(renders.get(best.candidate)!, padded.size, padded.offset, crop, [img.width, img.height], ex, o.logo);
  await writeFile(path.join(out, "marketing_plan.png"), final);
  await writeFile(path.join(out, "comparison.png"), await sideBySide(sourceCrop, final));
  report.seconds = (Date.now() - t0) / 1000;
  await writeFile(path.join(out, "report.json"), JSON.stringify(report, null, 2));
  console.log(`\nDone in ${report.seconds.toFixed(1)}s. Chose candidate ${best.candidate}${passing.length ? "" : "  (FORCED: did not pass checks)"}.`);
  console.log(`  ${path.join(out, "marketing_plan.png")}\n  ${path.join(out, "comparison.png")}  <- review this before using the image`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => fail(`Error: ${(e as Error).stack ?? e}`));
}
