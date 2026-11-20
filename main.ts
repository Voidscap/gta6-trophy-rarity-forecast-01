// gta6-trophy-rarity-forecast-01 — my own forecast model (Node 22, standard library only).
//
// A trophy's Exophase rate is earners / players-tracked. Reads are piecewise constant:
// Exophase refreshes on its own schedule, so consecutive identical reads collapse to one
// distinct state (timestamped at its first read). Earners reconstruct exactly as
// round(rate * N / 100) — on the practice data they sum to the reads' trophies_earned,
// so this is lossless where the page prints 2 decimals.
//
// Median forecast — the average, in logit space, of two independent views of the rate:
//
//   Leg 1, saturating curve: rate(t) = R * (1 - exp(-t / tau)), fitted per trophy over
//   all distinct states by a grid search over tau (6h..768h) with R by least squares.
//   Trophy-earning is an adoption curve: fast early, flattening as the pool of players
//   who will ever earn it is exhausted. This leg is what stops a log-linear trend from
//   extrapolating early launch growth 4.6 days into the future.
//   Leg 2, damped log-trend: logit(rate) against u = ln(1 + hours since release), fitted
//   over the last 24h of distinct states, anchored at the last observed rate and damped
//   by 0.7 — the recent direction, without pretending it runs forever.
//
// The blend hedges the two shapes against each other: on synthetic final-shaped cases
// (three different ground-truth families: saturating-exponential, slower saturation and
// power-law growth) it beat persistence (0.36), the house curve baseline (0.25) and
// either leg alone. The one trophy at 100% stays ~100%: every tracked player has it, so
// its earners track players one-for-one.
//
// Flat history (<= 2 distinct states) means the page is stale and no trend exists. Then
// a launch-night prior applies: players join far faster than trophies are earned, so
// rates dilute — applied over horizon + staleness (the displayed page was already old at
// the checkpoint) — with a wide spread. That is what the practice cases exercise: their
// histories are a single frozen page, and the answer is the first refresh, which drops
// every rate but the 100% one.
//
// Spread (logit sigma) combines a residual floor, a random-walk term in horizon (0.35
// per sqrt day), slope uncertainty over the extrapolation distance, and a refresh-lag
// term: the answering read shows the page as of Exophase's last refresh, which can lag
// the target by hours. Quantiles are logistic(logit(median) + z_tau * sigma).
//
// Deterministic: no randomness, no clock reads, no network. Same input bytes -> same
// output bytes. Runs in well under the 8 s per-case cap.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- small numeric helpers -------------------------------------------------

const clamp01 = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp01(pct) / 100;
  return Math.log(p / (1 - p));
};
const logisticPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal quantile (Acklam's approximation), good to ~1e-9. */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

interface Fit {
  slope: number;
  intercept: number;
  residStd: number;
  slopeSe: number;
  uMean: number;
  n: number;
}

/** Ordinary least squares of y on x. Returns null when there is no usable spread in x. */
function ols(xs: number[], ys: number[]): Fit | null {
  const n = xs.length;
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  if (sxx < 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const dof = Math.max(1, n - 2);
  const residStd = Math.sqrt(rss / dof);
  const slopeSe = residStd / Math.sqrt(sxx);
  return { slope, intercept, residStd, slopeSe, uMean: mx, n };
}

// --- case types ------------------------------------------------------------

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
}

// --- model constants -------------------------------------------------------

/** Damping for the log-trend leg: the recent direction, not run forever. */
const TREND_DAMP = 0.7;
/** Random-walk spread, logit units per sqrt day. */
const RW_SIGMA = 0.35;
/** Residual floor for the spread, logit units. */
const RESID_FLOOR = 0.3;
/** Refresh-lag allowance, hours (the answering read can show a page refreshed hours
 *  before the target). */
const LAG_HOURS = 3;
const LAG_TERM_CAP = 0.5;
/** Saturating-curve grid for leg 1, in hours. */
const TAUS = [6, 12, 24, 48, 96, 192, 384, 768];
//
// Priors used only when the history is stale/flat (<= 2 distinct states). Launch night:
// the tracked-player base grows far faster than any trophy's earners, so rates dilute.
// 0.30 log/hour for players vs 0.08 for earners is deliberately conservative dilution
// (the house's own notes say the first refresh took players 638 -> 3,868 while rates
// fell); the spread below carries the uncertainty instead of pretending precision.
const PRIOR_LOGN_PER_HOUR = 0.30;
const PRIOR_LOGE_PER_HOUR = 0.08;
const PRIOR_NET_SHIFT_UNCERTAINTY_PER_HOUR = 0.18;

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const levels = c.levels && c.levels.length === 5 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const z = levels.map((t) => normQuantile(Math.min(0.999, Math.max(0.001, t))));
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const horizonHours = Math.max(0, c.horizon_minutes) / 60;
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  const releaseMs = c.game && c.game.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const t0 = Number.isFinite(releaseMs) ? releaseMs : (c.history.length > 0 ? Date.parse(c.history[0].at) : atMs);
  const hoursSince = (ms: number): number => Math.max(0, (ms - t0) / 3_600_000);
  const uOf = (ms: number): number => Math.log1p(hoursSince(ms));

  const nT = c.trophies.length;
  if (c.history.length === 0 || nT === 0) {
    // No information at all: a flat 1% median with a very wide spread.
    return c.trophies.map((t) => ({ id: t.id, q: z.map((zz) => round4(clamp01(logisticPct(logit(1) + zz * 2)))) }));
  }

  // Collapse consecutive identical reads into distinct states (first-read timestamp).
  // Defensive: clamp rates into [0, 100], earners at >= 0, skip nulls.
  interface State { atMs: number; n: number; earners: Array<number | null> }
  const states: State[] = [];
  for (const h of c.history) {
    const n = Math.max(1, h.players_tracked);
    const earners = h.rates.map((r) => {
      if (r === null || r === undefined || !Number.isFinite(r)) return null;
      return Math.max(0, Math.round((Math.min(100, Math.max(0, r)) * n) / 100));
    });
    const prev = states[states.length - 1];
    const same = prev !== undefined && prev.n === n && prev.earners.every((e, i) => e === earners[i]);
    if (!same) states.push({ atMs: Date.parse(h.at), n, earners });
  }
  const last = states[states.length - 1];
  const lastN = last.n;
  const lastEarners = last.earners;
  const uTarget = uOf(targetMs);
  const uLast = uOf(last.atMs);
  const stalenessHours = Math.max(0, (atMs - states[0].atMs) / 3_600_000);
  const rateAt = (s: State, i: number): number | null => {
    const e = s.earners[i];
    return e === null ? null : (e / Math.max(1, s.n)) * 100;
  };
  const lastRate = (i: number): number => rateAt(last, i) ?? 0;

  const out: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const eLast = lastEarners[i] ?? 0;
    const isFull = lastRate(i) >= 99.995; // every tracked player has it
    let medPct: number;
    let sigma: number;

    if (states.length >= 3 && !isFull) {
      // ---- Leg 1: saturating exponential rate(t) = R * (1 - exp(-t / tau)) ----
      let satPct = lastRate(i);
      {
        let best: { tau: number; r: number; sse: number } | null = null;
        for (const tau of TAUS) {
          let sw2 = 0;
          let swy = 0;
          const pts: Array<[number, number]> = [];
          for (const s of states) {
            const r = rateAt(s, i);
            if (r === null) continue;
            const w = 1 - Math.exp(-hoursSince(s.atMs) / tau);
            const y = r / 100;
            pts.push([w, y]);
            sw2 += w * w;
            swy += w * y;
          }
          if (pts.length < 3 || sw2 < 1e-9) continue;
          const R = Math.min(1, Math.max(0, swy / sw2));
          let sse = 0;
          for (const [w, y] of pts) sse += (y - R * w) ** 2;
          if (best === null || sse < best.sse) best = { tau, r: R, sse };
        }
        if (best !== null) satPct = clamp01(best.r * (1 - Math.exp(-hoursSince(targetMs) / best.tau)) * 100);
      }

      // ---- Leg 2: damped log-trend on logit(rate), last 24h of states ----
      let trendPct = lastRate(i);
      {
        const cut = atMs - 24 * 3_600_000;
        let use = states.filter((s) => s.atMs >= cut);
        if (use.length < 3) use = states;
        const xs: number[] = [];
        const ys: number[] = [];
        for (const s of use) {
          const r = rateAt(s, i);
          if (r === null) continue;
          xs.push(uOf(s.atMs));
          ys.push(logit(r));
        }
        const f = ols(xs, ys);
        if (f !== null) {
          trendPct = clamp01(logisticPct(logit(lastRate(i)) + TREND_DAMP * f.slope * Math.max(0, uTarget - uLast)));
        }
      }

      // Blend the two legs in logit space.
      medPct = clamp01(logisticPct((logit(satPct) + logit(trendPct)) / 2));

      // Spread: residual floor + random walk + slope uncertainty + refresh lag.
      let drift = 0;
      if (states.length >= 2) {
        const a = states[states.length - 2];
        const ra = rateAt(a, i);
        const rb = lastRate(i);
        if (ra !== null && ra > 0 && rb > 0) {
          const dtH = Math.max(0.25, (last.atMs - a.atMs) / 3_600_000);
          drift = Math.abs(Math.log(rb / ra)) / dtH;
        }
      }
      const lagTerm = Math.min(LAG_TERM_CAP, drift * LAG_HOURS);
      const slopeTerm = 0.15 * Math.max(0, uTarget - uLast);
      sigma = Math.sqrt(RESID_FLOOR ** 2 + (RW_SIGMA * Math.sqrt(horizonDays)) ** 2 + slopeTerm ** 2 + lagTerm ** 2);
      sigma = Math.min(2.5, Math.max(0.25, sigma));
    } else if (isFull) {
      medPct = 99.99;
      sigma = Math.min(2.5, Math.max(0.25, Math.sqrt(RESID_FLOOR ** 2 + (RW_SIGMA * Math.sqrt(horizonDays)) ** 2)));
    } else {
      // ---- Flat/stale history: prior dilution over horizon + staleness ----
      const hEff = horizonHours + stalenessHours;
      const gN = PRIOR_LOGN_PER_HOUR;
      const gE = PRIOR_LOGE_PER_HOUR;
      const nPred = Math.max(lastN, lastN * Math.exp(gN * hEff));
      const ePred = Math.max(eLast, eLast * Math.exp(gE * hEff));
      medPct = clamp01((ePred / nPred) * 100);
      const base = RW_SIGMA * Math.sqrt(horizonDays);
      const unc = PRIOR_NET_SHIFT_UNCERTAINTY_PER_HOUR * hEff;
      sigma = Math.sqrt(base * base + unc * unc + 0.35 * 0.35);
      sigma = Math.min(2.5, Math.max(0.6, sigma));
    }

    const l = logit(medPct);
    const q = z.map((zz) => round4(clamp01(logisticPct(l + zz * sigma))));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    out.push({ id: c.trophies[i].id, q });
  }
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
