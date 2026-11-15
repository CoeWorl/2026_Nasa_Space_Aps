/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `npm run dev` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `npm run deploy` to publish your worker
 *
 * Bind resources to your worker in `wrangler.jsonc`. After adding bindings, a type definition for the
 * `Env` object can be regenerated with `npm run cf-typegen`.
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { fromUrl, type GeoTIFF, type GeoTIFFImage } from 'geotiff';
import './grounded-truth-react-app/src/styles/index.css';

type RGB = [number, number, number];
type IndexEntry = { name: string; label: string };

interface State {
	full: GeoTIFFImage | null;
	w: number;
	h: number;
	db: Float32Array;
}

const DEFAULT_API = import.meta.env.VITE_API_URL ?? 'https://groundtruth-api.groundtruth-api.workers.dev';

const $ = <T extends HTMLElement>(id: string): T => {
	const el = document.getElementById(id);
	if (!el) throw new Error(`Missing #${id}`);
	return el as T;
};

const el = {
	api: $<HTMLInputElement>('api'),
	scene: $<HTMLElement>('scene') as HTMLSelectElement,
	manual: $<HTMLDivElement>('manual'),
	name: $<HTMLInputElement>('name'),
	load: $<HTMLButtonElement>('load'),
	ramp: $<HTMLDivElement>('ramp'),
	lo: $<HTMLInputElement>('lo'),
	hi: $<HTMLInputElement>('hi'),
	loV: $<HTMLElement>('loV'),
	hiV: $<HTMLElement>('hiV'),
	readout: $<HTMLDivElement>('readout'),
	prov: $<HTMLElement>('prov'),
	stage: $<HTMLDivElement>('stage'),
	empty: $<HTMLDivElement>('empty'),
	status: $<HTMLDivElement>('status'),
	mark: $<HTMLDivElement>('mark'),
	cv: $<HTMLCanvasElement>('cv'),
};
const ctx = el.cv.getContext('2d')!;

let api = (new URLSearchParams(location.search).get('api') ?? DEFAULT_API).replace(/\/$/, '');
el.api.value = api;
el.api.addEventListener('change', () => {
	api = el.api.value.trim().replace(/\/$/, '');
	void init();
});

const state: State = { full: null, w: 0, h: 0, db: new Float32Array(0) };

/* ---------- colour ramp: low to high backscatter ---------- */

const STOPS: [number, RGB][] = [
	[0, [12, 20, 50]],
	[0.25, [22, 80, 120]],
	[0.5, [40, 150, 140]],
	[0.75, [230, 200, 90]],
	[1, [255, 250, 235]],
];

function ramp(t: number): RGB {
	const x = Math.min(1, Math.max(0, t));
	for (let i = 1; i < STOPS.length; i++) {
		const [b, cb] = STOPS[i];
		if (x <= b) {
			const [a, ca] = STOPS[i - 1];
			const f = (x - a) / (b - a);
			return [0, 1, 2].map((k) => ca[k] + (cb[k] - ca[k]) * f) as RGB;
		}
	}
	return STOPS[STOPS.length - 1][1];
}

el.ramp.style.background = 'linear-gradient(90deg,' + STOPS.map(([p, c]) => `rgb(${c.join(',')}) ${p * 100}%`).join(',') + ')';

/* ---------- helpers ---------- */

function setStatus(msg: string, isError = false): void {
	el.status.hidden = !msg;
	el.status.textContent = msg;
	el.status.className = isError ? 'err' : '';
}

function escapeHtml(s: string): string {
	return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** index.json may be an array or an object wrapping one; entries may be strings or objects. */
function entriesFrom(json: unknown): IndexEntry[] {
	const obj = json as Record<string, unknown>;
	const list: unknown[] = Array.isArray(json)
		? json
		: (((['items', 'files', 'artifacts', 'granules', 'scenes', 'data'] as const).map((k) => obj[k]).find(Array.isArray) as
				| unknown[]
				| undefined) ??
			(Object.values(obj).find(Array.isArray) as unknown[] | undefined) ??
			[]);

	const out: IndexEntry[] = [];
	for (const e of list) {
		const rec = (typeof e === 'object' && e !== null ? e : {}) as Record<string, unknown>;
		const raw = typeof e === 'string' ? e : String(rec.key ?? rec.path ?? rec.url ?? rec.file ?? rec.name ?? rec.filename ?? '');
		const name = raw.split('/').pop() ?? '';
		if (!/\.tiff?$/i.test(name)) continue;
		const tag = rec.date ?? rec.acquisition_time ?? rec.label;
		out.push({ name, label: tag ? `${String(tag)} (${name})` : name });
	}
	return out;
}

/** UTM to lat/lon on WGS84, northern hemisphere. */
function utmToLatLon(E: number, N: number, zone: number): [number, number] {
	const a = 6378137,
		f = 1 / 298.257223563,
		k0 = 0.9996;
	const e2 = f * (2 - f),
		ep2 = e2 / (1 - e2);
	const x = E - 500000;
	const mu = N / k0 / (a * (1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 ** 3) / 256));
	const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
	const p1 =
		mu +
		((3 * e1) / 2 - (27 * e1 ** 3) / 32) * Math.sin(2 * mu) +
		((21 * e1 ** 2) / 16 - (55 * e1 ** 4) / 32) * Math.sin(4 * mu) +
		((151 * e1 ** 3) / 96) * Math.sin(6 * mu);
	const s = Math.sin(p1),
		c = Math.cos(p1),
		t = Math.tan(p1);
	const C1 = ep2 * c * c,
		T1 = t * t;
	const N1 = a / Math.sqrt(1 - e2 * s * s);
	const R1 = (a * (1 - e2)) / Math.pow(1 - e2 * s * s, 1.5);
	const D = x / (N1 * k0);
	const lat = p1 - ((N1 * t) / R1) * ((D * D) / 2 - ((5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4) / 24);
	const lon = (D - ((1 + 2 * T1 + C1) * D ** 3) / 6 + ((5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5) / 120) / c;
	return [(lat * 180) / Math.PI, zone * 6 - 183 + (lon * 180) / Math.PI];
}

/* ---------- loading ---------- */

async function init(): Promise<void> {
	setStatus('Loading index…');
	try {
		const r = await fetch(`${api}/data/index.json`);
		if (!r.ok) throw new Error(`index.json returned ${r.status}`);
		const items = entriesFrom(await r.json());
		if (!items.length) throw new Error('index.json lists no GeoTIFFs');
		el.manual.hidden = true;
		el.scene.innerHTML = items.map((i) => `<option value="${escapeHtml(i.name)}">${escapeHtml(i.label)}</option>`).join('');
		el.scene.onchange = () => void openScene(el.scene.value);
		setStatus('');
		await openScene(items[0].name);
	} catch (e) {
		el.manual.hidden = false;
		el.scene.innerHTML = '';
		el.empty.hidden = false;
		el.empty.innerHTML = '<p>Could not load the observation list. Check the Worker URL, or enter an artifact name.</p>';
		setStatus(errMsg(e), true);
	}
}

el.load.onclick = () => {
	const n = el.name.value.trim();
	if (n) void openScene(n);
};

async function openScene(name: string): Promise<void> {
	setStatus(`Opening ${name}…`);
	try {
		const tiff: GeoTIFF = await fromUrl(`${api}/data/${name}`);
		const count = await tiff.getImageCount();
		const full = await tiff.getImage(0);

		// Smallest overview at least ~1000 px wide keeps the display read cheap.
		let disp = full;
		for (let i = count - 1; i >= 0; i--) {
			const im = await tiff.getImage(i);
			if (im.getWidth() >= 1000) {
				disp = im;
				break;
			}
		}

		const w = disp.getWidth(),
			h = disp.getHeight();
		const bands = (await disp.readRasters({ samples: [0], interleave: false })) as unknown as ArrayLike<number>[];
		const band = bands[0];

		const db = new Float32Array(w * h);
		const sample: number[] = [];
		for (let i = 0; i < band.length; i++) {
			const v = band[i];
			db[i] = v > 0 ? 10 * Math.log10(v) : NaN;
			if (!Number.isNaN(db[i]) && i % 7 === 0) sample.push(db[i]);
		}
		if (!sample.length) throw new Error('No valid pixels in this raster.');
		sample.sort((a, b) => a - b);
		const pct = (q: number) => sample[Math.floor(q * (sample.length - 1))];

		state.full = full;
		state.w = w;
		state.h = h;
		state.db = db;
		for (const s of [el.lo, el.hi]) {
			s.min = String(Math.floor(sample[0]));
			s.max = String(Math.ceil(sample[sample.length - 1]));
		}
		el.lo.value = String(Math.round(pct(0.02)));
		el.hi.value = String(Math.round(pct(0.98)));

		el.cv.width = w;
		el.cv.height = h;
		el.stage.hidden = false;
		el.empty.hidden = true;
		el.mark.style.display = 'none';
		el.readout.innerHTML = '<p>No pixel selected.</p>';
		draw();
		setStatus('');
		void loadProvenance(name);
	} catch (e) {
		setStatus(errMsg(e), true);
	}
}

function draw(): void {
	const lo = Number(el.lo.value);
	let hi = Number(el.hi.value);
	if (hi <= lo) hi = lo + 0.5;
	el.loV.textContent = `${lo.toFixed(1)} dB`;
	el.hiV.textContent = `${hi.toFixed(1)} dB`;

	const img = ctx.createImageData(state.w, state.h);
	const d = img.data;
	for (let i = 0; i < state.db.length; i++) {
		const v = state.db[i];
		if (Number.isNaN(v)) continue; // nodata stays transparent
		const c = ramp((v - lo) / (hi - lo));
		d[i * 4] = c[0];
		d[i * 4 + 1] = c[1];
		d[i * 4 + 2] = c[2];
		d[i * 4 + 3] = 255;
	}
	ctx.putImageData(img, 0, 0);
}
el.lo.oninput = el.hi.oninput = () => {
	if (state.full) draw();
};

/* ---------- click to inspect ---------- */

el.cv.addEventListener('click', async (ev: MouseEvent) => {
	const full = state.full;
	if (!full) return;
	const r = el.cv.getBoundingClientRect();
	const fx = (ev.clientX - r.left) / r.width;
	const fy = (ev.clientY - r.top) / r.height;
	el.mark.style.display = 'block';
	el.mark.style.left = `${fx * r.width + el.cv.offsetLeft}px`;
	el.mark.style.top = `${fy * r.height + el.cv.offsetTop}px`;

	const W = full.getWidth(),
		H = full.getHeight();
	const px = Math.min(W - 1, Math.floor(fx * W));
	const py = Math.min(H - 1, Math.floor(fy * H));

	try {
		// One-pixel window at full resolution: the Worker serves it as a tiny range request.
		const bands = (await full.readRasters({
			window: [px, py, px + 1, py + 1],
			samples: [0],
			interleave: false,
		})) as unknown as ArrayLike<number>[];
		const v = bands[0][0];

		const [ox, oy] = full.getOrigin();
		const [rx, ry] = full.getResolution();
		const E = ox + (px + 0.5) * rx,
			N = oy + (py + 0.5) * ry;
		const epsg = Number(full.getGeoKeys()?.ProjectedCSTypeGeoKey ?? 0);

		let where = `E ${E.toFixed(0)}  N ${N.toFixed(0)} m`;
		if (epsg >= 32601 && epsg <= 32660) {
			const [lat, lon] = utmToLatLon(E, N, epsg - 32600);
			where = `${lat.toFixed(5)}°, ${lon.toFixed(5)}° · ${where}`;
		}

		el.readout.innerHTML = !(v > 0)
			? `<div class="big">No data</div><p>Pixel ${px}, ${py} is masked or outside the footprint.</p><p>${where}</p>`
			: `<div class="big">${(10 * Math.log10(v)).toFixed(2)} <small>dB</small></div>` +
				`<p>Linear ${v.toPrecision(6)}</p><p>Pixel ${px}, ${py}</p><p>${where}</p>`;
	} catch (e) {
		el.readout.innerHTML = `<p class="error">${escapeHtml(errMsg(e))}</p>`;
	}
});

/* ---------- provenance ---------- */

async function loadProvenance(name: string): Promise<void> {
	try {
		const r = await fetch(`${api}/data/${name.replace(/\.tiff?$/i, '')}.provenance.json`);
		if (!r.ok) throw new Error();
		const j = (await r.json()) as Record<string, unknown>;
		el.prov.innerHTML = Object.entries(j)
			.map(
				([k, v]) =>
					`<dt>${escapeHtml(k.replace(/_/g, ' '))}</dt><dd>${escapeHtml(typeof v === 'object' ? JSON.stringify(v) : String(v))}</dd>`,
			)
			.join('');
	} catch {
		el.prov.innerHTML = '<dt>Status</dt><dd>No provenance file for this observation.</dd>';
	}
}

void init();
