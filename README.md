# Marketing Floor Plan

Turns a source floor plan (CAD plan image) into a clean, marketing-ready floor plan using Gemini, written in TypeScript.

## Pipeline

1. **Input check** – resolution check, and Gemini classifies the input (CAD plan / photo / sketch / multi-unit).
2. **Extraction** – Gemini returns the plan as JSON (rooms, sizes, doors, windows, open-plan links, exterior zones, furniture, ambiguous symbols, text-label boxes).
3. **Confirmation** – review the extraction; edit `extraction.json` and rerun with `--extraction` if needed.
4. **Cleaning** – dimension chains, labels, tags and compass are removed while walls stay intact.
5. **Generation** – N candidates from the Gemini image model with a locked prompt (no text, no people).
6. **Verification** – Gemini compares each candidate with the source; layout errors fail the candidate.
7. **Overlay** – room names, sizes, compass, disclaimer and optional logo are drawn in code (SVG).
8. **Review output** – side-by-side comparison plus `report.json`. If no candidate passes, no final image is produced.

## Setup

```bash
npm install
export GEMINI_API_KEY=your_key_here
```

## Usage

```bash
npx tsx floorplan-marketing.ts plan.png
npx tsx floorplan-marketing.ts plan.png --candidates 4 --logo logo.png --yes
npx tsx floorplan-marketing.ts plan.png --extraction out/plan/extraction.json
```

Other scripts: `npm start`, `npm run typecheck`.

## Configuration

Model versions can be overridden with `FP_TEXT_MODEL` and `FP_IMAGE_MODEL`. Pin exact versions and rerun your test set before changing them.
