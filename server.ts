/**
 * Web UI for the marketing floor plan pipeline.
 * Upload a plan -> runs floorplan-marketing.ts as a job -> shows progress and the result.
 *   npm run ui   (then open http://localhost:3000)
 */
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";

const PORT = Number(process.env.PORT ?? 3000);
const JOBS_DIR = path.resolve("jobs");
const MAX_UPLOAD = 25 * 1024 * 1024;
const EXTS = new Set([".png", ".jpg", ".jpeg", ".webp"]);

type Job = { id: string; status: "running" | "done" | "failed"; log: string; code: number | null };
const jobs = new Map<string, Job>();

const PAGE = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marketing Floor Plan</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap">
<style>
 :root{--primary:#1976d2;--primary-dark:#115293;--primary-light:#4791db;--text:rgba(0,0,0,.87);--text2:rgba(0,0,0,.6);--divider:rgba(0,0,0,.12);
  --e1:0 2px 1px -1px rgba(0,0,0,.2),0 1px 1px 0 rgba(0,0,0,.14),0 1px 3px 0 rgba(0,0,0,.12);
  --e2:0 3px 1px -2px rgba(0,0,0,.2),0 2px 2px 0 rgba(0,0,0,.14),0 1px 5px 0 rgba(0,0,0,.12);
  --e4:0 2px 4px -1px rgba(0,0,0,.2),0 4px 5px 0 rgba(0,0,0,.14),0 1px 10px 0 rgba(0,0,0,.12)}
 *{box-sizing:border-box}
 body{margin:0;background:#fafafa;color:var(--text);font:400 1rem/1.5 Roboto,"Helvetica Neue",Arial,sans-serif;letter-spacing:.00938em}
 header{background:var(--primary);color:#fff;box-shadow:var(--e4);height:64px;display:flex;align-items:center;padding:0 24px}
 header h1{margin:0;font:500 1.25rem/1.6 Roboto,sans-serif;letter-spacing:.0075em}
 main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
 .sub{color:var(--text2);margin:0 0 8px}
 .paper{background:#fff;border-radius:4px;box-shadow:var(--e1);padding:24px;margin:16px 0}
 h2{margin:0 0 16px;font:400 1.5rem/1.334 Roboto,sans-serif}
 .field{position:relative;margin:24px 0 0}
 .field>label.fl{position:absolute;top:-9px;left:10px;padding:0 5px;background:#fff;font-size:.75rem;line-height:1.2;color:var(--text2);z-index:1;pointer-events:none}
 .field:focus-within>label.fl{color:var(--primary)}
 select{appearance:none;-webkit-appearance:none;width:100%;max-width:340px;display:block;font:inherit;color:var(--text);padding:16.5px 40px 16.5px 14px;
  border:1px solid rgba(0,0,0,.23);border-radius:4px;background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath fill='rgba(0,0,0,.54)' d='M7 10l5 5 5-5z'/%3E%3C/svg%3E") no-repeat right 8px center;cursor:pointer}
 select:hover{border-color:var(--text)}
 select:focus{outline:0;border-color:var(--primary);box-shadow:inset 0 0 0 1px var(--primary)}
 .hint{font-size:.75rem;color:var(--text2);margin:4px 14px 0}
 .filebox{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:8px}
 .filebox input[type=file]{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;clip:rect(0 0 0 0)}
 .filename{color:var(--text2);font-size:.875rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%}
 .cap{font-size:.875rem;font-weight:500;color:var(--text);margin:24px 0 0}
 #preview{display:none;max-height:200px;margin-top:16px}
 img{max-width:100%;border-radius:4px;box-shadow:var(--e1)}
 .btn{position:relative;overflow:hidden;display:inline-flex;align-items:center;justify-content:center;gap:8px;min-width:64px;padding:6px 16px;border:0;border-radius:4px;
  font:500 .875rem/1.75 Roboto,sans-serif;letter-spacing:.02857em;text-transform:uppercase;text-decoration:none;cursor:pointer;user-select:none;
  transition:background-color .25s,box-shadow .25s,border-color .25s}
 .btn:focus-visible{outline:2px solid var(--primary-light);outline-offset:2px}
 .btn.contained{background:var(--primary);color:#fff;box-shadow:var(--e2)}
 .btn.contained:hover{background:var(--primary-dark);box-shadow:var(--e4)}
 .btn.contained[disabled]{background:rgba(0,0,0,.12);color:rgba(0,0,0,.26);box-shadow:none;cursor:default;pointer-events:none}
 .btn.outlined{background:transparent;color:var(--primary);border:1px solid rgba(25,118,210,.5);padding:5px 15px}
 .btn.outlined:hover{background:rgba(25,118,210,.04);border-color:var(--primary)}
 .btn.lg{padding:8px 22px;font-size:.9375rem}
 .btn svg{width:20px;height:20px;fill:currentColor;flex:none}
 .ripple{position:absolute;border-radius:50%;background:currentColor;opacity:.3;transform:scale(0);animation:rip .6s ease-out;pointer-events:none}
 @keyframes rip{to{transform:scale(1);opacity:0}}
 .actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
 .alert{display:flex;align-items:center;gap:12px;padding:6px 16px;border-radius:4px;font-size:.875rem;line-height:1.43;min-height:48px;background:#e5f6fd;color:#014361}
 .alert.success{background:#edf7ed;color:#1e4620}.alert.error{background:#fdeded;color:#5f2120;font-weight:500}
 .progress{height:4px;margin-top:12px;border-radius:2px;background:#bbdefb;overflow:hidden;position:relative}
 .progress::before{content:"";position:absolute;top:0;bottom:0;left:0;width:40%;background:var(--primary);border-radius:2px;animation:ind 1.6s cubic-bezier(.65,.81,.73,.4) infinite}
 .progress.off{display:none}
 @keyframes ind{0%{left:-40%}100%{left:100%}}
 pre{background:#212121;color:#e0e0e0;font:.8125rem/1.5 "Roboto Mono",ui-monospace,Menlo,monospace;padding:16px;border-radius:4px;max-height:300px;overflow:auto;white-space:pre-wrap;margin:16px 0 0}
 pre:empty{display:none}
 .note{color:var(--text2);margin:16px 0 0;font-size:.875rem}
</style>
<header><h1>Marketing Floor Plan</h1></header>
<main>
<p class="sub">Upload a CAD floor plan of a single unit. It is converted to a clean, marketing-ready plan using Gemini.</p>
<form id="f" class="paper">
 <p class="cap" style="margin-top:0">Floor plan image</p>
 <div class="filebox">
  <label class="btn outlined"><svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>Choose image<input type="file" id="file" accept=".png,.jpg,.jpeg,.webp,image/*" required></label>
  <span class="filename" id="fname">No file selected</span>
 </div>
 <div class="hint" style="margin-left:0">PNG, JPG or WebP; at least 800px on the short side</div>
 <img id="preview" alt="">
 <div class="field"><label class="fl" for="cand">Candidates</label>
  <select id="cand"><option>1</option><option selected>3</option><option>4</option><option>6</option></select>
  <div class="hint">More = better odds of a pass, higher API cost</div></div>
 <div class="field"><label class="fl" for="size">Resolution</label>
  <select id="size"><option>1K</option><option selected>2K</option><option>4K</option></select></div>
 <p class="cap">Logo (optional)</p>
 <div class="filebox">
  <label class="btn outlined"><svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>Choose logo<input type="file" id="logo" accept="image/*"></label>
  <span class="filename" id="lname">No file selected</span>
 </div>
 <div class="actions" style="margin-top:32px"><button class="btn contained lg" id="go">Convert</button></div>
</form>
<div id="out"></div>
</main>
<script>
const $=id=>document.getElementById(id);
document.addEventListener("pointerdown",e=>{
 const b=e.target.closest(".btn");if(!b||b.disabled)return;
 const r=b.getBoundingClientRect(),d=Math.max(r.width,r.height)*2,s=document.createElement("span");
 s.className="ripple";s.style.cssText="width:"+d+"px;height:"+d+"px;left:"+(e.clientX-r.left-d/2)+"px;top:"+(e.clientY-r.top-d/2)+"px";
 b.appendChild(s);s.onanimationend=()=>s.remove();
});
$("file").onchange=e=>{const f=e.target.files[0];$("fname").textContent=f?f.name:"No file selected";if(f){$("preview").src=URL.createObjectURL(f);$("preview").style.display="block"}};
$("logo").onchange=e=>{const f=e.target.files[0];$("lname").textContent=f?f.name:"No file selected"};
const readB64=f=>new Promise(r=>{const x=new FileReader();x.onload=()=>r(x.result.split(",")[1]);x.readAsDataURL(f)});
$("f").onsubmit=async e=>{
 e.preventDefault(); $("go").disabled=true;
 const f=$("file").files[0], lg=$("logo").files[0];
 $("out").innerHTML='<div class="paper"><div class="alert" id="st">Uploading...</div><div class="progress" id="pg"></div><pre id="log"></pre></div>';
 const body={name:f.name,data:await readB64(f),candidates:$("cand").value,size:$("size").value};
 if(lg){body.logoName=lg.name;body.logoData=await readB64(lg)}
 const r=await fetch("/api/jobs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
 const j=await r.json();
 if(!r.ok){$("out").innerHTML='<div class="paper"><div class="alert error"></div></div>';$("out").querySelector(".alert").textContent=j.error;$("go").disabled=false;return}
 poll(j.id);
};
async function poll(id){
 const j=await (await fetch("/api/jobs/"+id)).json();
 $("log").textContent=j.log; $("log").scrollTop=1e9;
 if(j.status==="running"){$("st").textContent="Converting... this takes a few minutes";return setTimeout(()=>poll(id),2000)}
 $("go").disabled=false;
 $("pg").classList.add("off");
 if(j.status==="failed"){
  $("st").className="alert error";$("st").textContent="Could not convert this plan. See the log below.";
  return;
 }
 $("st").className="alert success";$("st").textContent="Done";
 const u=n=>"/files/"+id+"/"+n;
 $("out").insertAdjacentHTML("beforeend",
  '<div class="paper"><h2>Marketing plan</h2><img src="'+u("marketing_plan.png")+'">'+
  '<div class="actions"><a class="btn contained" href="'+u("marketing_plan.png")+'" download>Download marketing plan</a>'+
  '<a class="btn outlined" href="'+u("comparison.png")+'" target="_blank">Side-by-side comparison</a></div>'+
  '<p class="note">Review the comparison against your source before using the image.</p></div>');
}
</script>`;

function send(res: http.ServerResponse, code: number, body: string | Buffer, type = "application/json") {
  res.writeHead(code, { "content-type": type });
  res.end(body);
}

async function readBody(req: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of req as AsyncIterable<Buffer>) {
    n += c.length;
    if (n > MAX_UPLOAD * 2) throw new Error("Upload too large");
    chunks.push(c);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function startJob(req: http.IncomingMessage, res: http.ServerResponse) {
  let b: any;
  try {
    b = JSON.parse(await readBody(req));
  } catch {
    return send(res, 400, JSON.stringify({ error: "Invalid or too-large upload." }));
  }
  const ext = path.extname(String(b.name ?? "")).toLowerCase();
  if (!EXTS.has(ext) || typeof b.data !== "string") return send(res, 400, JSON.stringify({ error: "Please upload a PNG, JPG or WebP image." }));
  const image = Buffer.from(b.data, "base64");
  if (!image.length || image.length > MAX_UPLOAD) return send(res, 400, JSON.stringify({ error: "Image is empty or larger than 25 MB." }));
  if (!["1K", "2K", "4K"].includes(b.size) || !/^\d+$/.test(String(b.candidates)) || +b.candidates < 1 || +b.candidates > 10)
    return send(res, 400, JSON.stringify({ error: "Invalid options." }));

  const id = randomUUID();
  const dir = path.join(JOBS_DIR, id);
  await mkdir(dir, { recursive: true });
  const input = path.join(dir, `plan${ext}`);
  await writeFile(input, image);
  const args = ["tsx", "floorplan-marketing.ts", input, "--out", path.join(dir, "out"), "--candidates", String(b.candidates), "--size", b.size, "--yes"];
  const logoExt = path.extname(String(b.logoName ?? "")).toLowerCase();
  if (typeof b.logoData === "string" && EXTS.has(logoExt)) {
    const logo = path.join(dir, `logo${logoExt}`);
    await writeFile(logo, Buffer.from(b.logoData, "base64"));
    args.push("--logo", logo);
  }

  const job: Job = { id, status: "running", log: "", code: null };
  jobs.set(id, job);
  const child = spawn("npx", args, { env: process.env });
  const add = (d: Buffer) => (job.log += d.toString());
  child.stdout.on("data", add);
  child.stderr.on("data", add);
  child.on("error", (e) => { job.log += `\n${e.message}`; job.status = "failed"; });
  child.on("close", (code) => {
    job.code = code;
    job.status = code === 0 && existsSync(path.join(dir, "out", "marketing_plan.png")) ? "done" : "failed";
  });
  send(res, 202, JSON.stringify({ id }));
}

const UUID = /^[0-9a-f-]{36}$/;

http
  .createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://x");
      if (req.method === "GET" && url.pathname === "/") return send(res, 200, PAGE, "text/html; charset=utf-8");
      if (req.method === "POST" && url.pathname === "/api/jobs") return await startJob(req, res);
      let m = url.pathname.match(/^\/api\/jobs\/([^/]+)$/);
      if (req.method === "GET" && m) {
        const job = jobs.get(m[1]);
        if (!job) return send(res, 404, JSON.stringify({ error: "Unknown job" }));
        return send(res, 200, JSON.stringify({ status: job.status, log: job.log, files: [] }));
      }
      m = url.pathname.match(/^\/files\/([^/]+)\/(marketing_plan|comparison)\.png$/);
      if (req.method === "GET" && m && UUID.test(m[1])) {
        const file = path.join(JOBS_DIR, m[1], "out", `${m[2]}.png`);
        if (!existsSync(file)) return send(res, 404, "Not found", "text/plain");
        return send(res, 200, await readFile(file), "image/png");
      }
      send(res, 404, "Not found", "text/plain");
    } catch (e) {
      send(res, 500, JSON.stringify({ error: String(e) }));
    }
  })
  .listen(PORT, () => console.log(`Marketing Floor Plan UI: http://localhost:${PORT}`));
