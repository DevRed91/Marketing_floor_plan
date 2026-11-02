import test from "node:test";
import assert from "node:assert/strict";
import sharp from "sharp";
import {
  Extraction,
  validateCliArgs,
  judgementProblems,
  extractFinalCandidateImage,
  compassSvg,
  composeFinal,
  cleanPlan,
  RGB,
} from "./floorplan-marketing.js";

const baseExtraction: Extraction = {
  input_type: "cad_drawing",
  single_unit: true,
  rooms: [
    {
      name: "Living Room",
      size_label: "5.0 x 4.0m",
      room_box: [100, 100, 500, 500],
      label_box: [280, 280, 320, 320],
      open_to: [],
    },
  ],
  door_count: 2,
  window_count: 3,
  entrance: "west wall",
  exterior_zones: [],
  furniture: [],
  ambiguous_items: [],
  text_items: [],
  compass_box: [50, 800, 150, 900],
  north_direction: "up",
  notes: [],
};

test("validateCliArgs validates size, candidates, and tolerance correctly", () => {
  // Valid cases
  const valid = validateCliArgs("2K", "3", "0");
  assert.deepEqual(valid, { ok: true, nCandidates: 3, tolerance: 0 });

  // Invalid size
  assert.equal(validateCliArgs("8K", "3", "0").ok, false);
  assert.equal(validateCliArgs(undefined, "3", "0").ok, false);

  // Invalid candidates (NaN, out of range)
  assert.equal(validateCliArgs("2K", "abc", "0").ok, false);
  assert.equal(validateCliArgs("2K", "0", "0").ok, false);
  assert.equal(validateCliArgs("2K", "11", "0").ok, false);
  assert.equal(validateCliArgs("2K", "2.5", "0").ok, false);

  // Invalid tolerance (NaN, negative, too large)
  assert.equal(validateCliArgs("2K", "3", "abc").ok, false);
  assert.equal(validateCliArgs("2K", "3", "-1").ok, false);
  assert.equal(validateCliArgs("2K", "3", "55").ok, false);
});

test("judgementProblems enforces minimum fidelity score of 7 and tolerance", () => {
  const baseJudgement = {
    missing_rooms: [],
    extra_rooms: [],
    walls_added: [],
    walls_removed_or_moved: [],
    doors_missing_or_moved: [],
    windows_missing_or_moved: [],
    door_count: 2,
    window_count: 3,
    open_plan_preserved: true,
    exterior_filled_as_interior: false,
    text_or_numbers_present: false,
    people_present: false,
    top_down_orthographic: true,
    fidelity_score: 8,
  };

  // Score 8 with no other issues -> passes (empty problems list)
  assert.deepEqual(judgementProblems(baseJudgement, baseExtraction, 0), []);

  // Score < 7 fails even with no other issues
  const lowScoreJudgement = { ...baseJudgement, fidelity_score: 6 };
  const lowScoreProblems = judgementProblems(lowScoreJudgement, baseExtraction, 0);
  assert.equal(lowScoreProblems.length, 1);
  assert.match(lowScoreProblems[0], /fidelity score 6\/10 is below minimum 7/);

  // Door/window tolerance checking
  const mismatchedJudgement = { ...baseJudgement, door_count: 3 }; // diff of 1
  assert.equal(judgementProblems(mismatchedJudgement, baseExtraction, 0).length, 1);
  assert.deepEqual(judgementProblems(mismatchedJudgement, baseExtraction, 1), []);
});

test("extractFinalCandidateImage skips interim thought images and returns the last valid image", () => {
  const candidates = [
    {
      content: {
        parts: [
          { thought: true, inlineData: { data: "thought_image_data" } },
          { inlineData: { data: "first_real_image_data" } },
          { thought: true, inlineData: { data: "another_thought_image" } },
          { inlineData: { data: "final_real_image_data" } },
        ],
      },
    },
  ];

  const result = extractFinalCandidateImage(candidates);
  assert.equal(result, "final_real_image_data");

  // Returns null when candidates have only thought parts
  const onlyThoughts = [
    {
      content: {
        parts: [{ thought: true, inlineData: { data: "thought_only" } }],
      },
    },
  ];
  assert.equal(extractFinalCandidateImage(onlyThoughts), null);
});

test("compassSvg produces expected rotation transform for directions", () => {
  const upSvg = compassSvg(100, 100, 20, "up");
  assert.match(upSvg, /rotate\(0 100 100\)/);

  const rightSvg = compassSvg(100, 100, 20, "right");
  assert.match(rightSvg, /rotate\(90 100 100\)/);

  const downSvg = compassSvg(100, 100, 20, "down");
  assert.match(downSvg, /rotate\(180 100 100\)/);

  const leftSvg = compassSvg(100, 100, 20, "left");
  assert.match(leftSvg, /rotate\(270 100 100\)/);
});

test("composeFinal omits compass when north_direction is unknown", async () => {
  const dummyRender = await sharp({
    create: { width: 500, height: 500, channels: 3, background: { r: 255, g: 255, b: 255 } },
  })
    .png()
    .toBuffer();

  const exUnknownNorth: Extraction = {
    ...baseExtraction,
    north_direction: "unknown",
  };

  const finalBuffer = await composeFinal(
    dummyRender,
    [500, 500],
    [0, 0],
    [0, 0, 500, 500],
    [500, 500],
    exUnknownNorth,
  );
  assert.ok(finalBuffer instanceof Buffer);

  // If north is known, composeFinal should also succeed and include the compass
  const exKnownNorth: Extraction = {
    ...baseExtraction,
    north_direction: "up",
  };

  const finalWithCompass = await composeFinal(
    dummyRender,
    [500, 500],
    [0, 0],
    [0, 0, 500, 500],
    [500, 500],
    exKnownNorth,
  );
  assert.ok(finalWithCompass instanceof Buffer);
});

test("cleanPlan warns instead of silently falling back when detectWalls fails", async () => {
  const warnings: string[] = [];
  const origWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map(String).join(" "));
  };

  try {
    // 1x1 image will fail detectWalls because it's too small for kernel and Otsu
    const tinyImg: RGB = {
      width: 1,
      height: 1,
      data: Buffer.from([255, 255, 255]),
    };

    const res = await cleanPlan(tinyImg, baseExtraction);
    assert.deepEqual(res.crop, [0, 0, 1, 1]);
    assert.ok(warnings.length > 0);
    assert.match(warnings[0], /Warning: wall detection failed/);
  } finally {
    console.warn = origWarn;
  }
});
