/*
 * Hemodynamic model of Buxton, Uludağ, Dubowitz & Liu (2004), NeuroImage 23, S220–S233.
 * Stimulus -> neural response (Eq. 14) -> CBF / CMRO2 (Eqs. 12-13) -> balloon model (Eqs. 10-11)
 * -> BOLD (Eq. 9). A line-for-line port of scripts/balloon_model_interactive.py (proj_DystoniaHRF);
 * see that file's docstring for the modelling choices.
 */
(function (root) {
  "use strict";

  const K_GAMMA = 3;        // gamma-variate order k (Eq. 12)
  const TAU_F = 4.0;        // τ_f: FWHM of the CBF impulse response h(t) [s]
  const TAU_M = 4.0;        // τ_m: FWHM of the CMRO2 impulse response g(t) [s]
  const DELTA_T_M = 1.0;    // δt_m: delay of the CMRO2 impulse response [s]
  const A1 = 3.4, A2 = 1.0; // BOLD weights a_1, a_2 (Obata et al., 2004; 1.5 T, TE = 40 ms)
  const ONSET = 5.0;        // first stimulus onset [s]

  // Table-1 / Fig. 4A defaults.
  const DEFAULTS = {
    D: 20, ISI: 1, kappa: 0, tau_I: 2, f1: 1.5, delta_t: 0, n: 3,
    tau_plus: 0, tau_minus: 0, alpha: 0.4, tau_MTT: 3, V0: 0.03,
  };

  // Fig. 4 timing variants: A τ± = 0; B τ+ = τ− = 20 s (caption); C τ− = 20 s, δt = 1 s.
  const FIG4 = {
    A: { tau_plus: 0, tau_minus: 0, delta_t: 0 },
    B: { tau_plus: 20, tau_minus: 20, delta_t: 0 },
    C: { tau_plus: 0, tau_minus: 20, delta_t: 1 },
  };

  /** Stimulus events [[onset, duration], ...] and simulated duration for a pattern. */
  function stimulusEvents(pattern, p) {
    if (pattern === "pair") {          // two 1-s events, onsets 1 s + ISI apart
      return { events: [[ONSET, 1], [ONSET + 1 + p.ISI, 1]], tMax: 60 };
    }
    if (pattern === "fig5") {          // Fig. 5: single, single 20 s later, pair (1-s ISI), 20-s block
      return { events: [[5, 1], [25, 1], [45, 1], [47, 1], [65, 20]], tMax: 110 };
    }
    return { events: [[ONSET, p.D]], tMax: Math.max(60, ONSET + p.D + 35) };
  }

  function timeGrid(tMax, dt) {
    const n = Math.round(tMax / dt);
    return Float64Array.from({ length: n }, (_, i) => i * dt);
  }

  /** s(t): 1 during any event, else 0. */
  function stimulus(t, events) {
    return t.map((ti) => (events.some(([on, dur]) => ti >= on && ti < on + dur) ? 1 : 0));
  }

  /** N(t), Eq. 14: N = max(s − I, 0), dI/dt = (κN − I)/τ_I, RK4. */
  function neuralResponse(s, kappa, tauI, dt) {
    const dIdt = (I, sv) => (kappa * Math.max(sv - I, 0) - I) / tauI;
    const I = new Float64Array(s.length);
    for (let i = 0; i < s.length - 1; i++) {
      const s0 = s[i], s1 = s[i + 1], sm = 0.5 * (s0 + s1);
      const k1 = dIdt(I[i], s0);
      const k2 = dIdt(I[i] + 0.5 * dt * k1, sm);
      const k3 = dIdt(I[i] + 0.5 * dt * k2, sm);
      const k4 = dIdt(I[i] + dt * k3, s1);
      I[i + 1] = I[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return s.map((sv, i) => Math.max(sv - I[i], 0));
  }

  /** Unit-area gamma variate, Eq. 12: (t/τ_h)^k e^(−t/τ_h) / [k τ_h (k−1)!], τ_h = 0.242·FWHM. */
  function gammaKernel(n, dt, fwhm, k = K_GAMMA) {
    const tauH = 0.242 * fwhm;
    let fact = 1;
    for (let j = 2; j < k; j++) fact *= j;
    const norm = k * tauH * fact;
    return Float64Array.from({ length: n }, (_, i) => {
      const x = (i * dt) / tauH;
      return (x ** k * Math.exp(-x)) / norm;
    });
  }

  /** Causal kernel(t − delay) * N(t) (Riemann sum), the convolution of Eq. 13. */
  function convolveDelayed(N, kernel, delay, dt) {
    const n = N.length, out = new Float64Array(n), shift = Math.round(delay / dt);
    const kLen = Math.min(kernel.length, n);
    for (let i = shift; i < n; i++) {
      const j = i - shift;
      let acc = 0;
      for (let k = 0; k < kLen && k <= j; k++) acc += N[j - k] * kernel[k];
      out[i] = acc * dt;
    }
    return out;
  }

  /** Balloon model, Eqs. 10-11: dv/dt = [f − v^(1/α)]/(τ_MTT + τ), τ = τ+ inflating / τ− deflating;
   *  dq/dt = [m − (q/v) f_out]/τ_MTT with f_out = v^(1/α) + τ dv/dt and f E/E0 = m (Eq. 2). RK4. */
  function balloon(f, m, tauMTT, tauPlus, tauMinus, alpha, dt) {
    const invA = 1 / alpha, n = f.length;
    const q = new Float64Array(n).fill(1), v = new Float64Array(n).fill(1);
    const derivs = (qq, vv, fv, mv) => {
      vv = Math.max(vv, 1e-6);
      const fss = vv ** invA, drive = fv - fss;
      const tau = drive >= 0 ? tauPlus : tauMinus;
      const dv = drive / (tauMTT + tau);
      return [(mv - (qq / vv) * (fss + tau * dv)) / tauMTT, dv];
    };
    for (let i = 0; i < n - 1; i++) {
      const fm = 0.5 * (f[i] + f[i + 1]), mm = 0.5 * (m[i] + m[i + 1]);
      const q0 = q[i], v0 = v[i];
      const [a1, b1] = derivs(q0, v0, f[i], m[i]);
      const [a2, b2] = derivs(q0 + 0.5 * dt * a1, v0 + 0.5 * dt * b1, fm, mm);
      const [a3, b3] = derivs(q0 + 0.5 * dt * a2, v0 + 0.5 * dt * b2, fm, mm);
      const [a4, b4] = derivs(q0 + dt * a3, v0 + dt * b3, f[i + 1], m[i + 1]);
      q[i + 1] = q0 + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
      v[i + 1] = v0 + (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    }
    return { q, v };
  }

  /** BOLD signal change [%], Eq. 9: V0 [a1 (1 − q) − a2 (1 − v)]. */
  function bold(q, v, V0) {
    return q.map((qi, i) => 100 * V0 * (A1 * (1 - qi) - A2 * (1 - v[i])));
  }

  /**
   * Run the full chain on time grid t for stimulus events and parameters p.
   * opts.variants: also return the Fig. 4 timing variants (b, q, v) with every other parameter from p.
   */
  function simulate(t, events, p, dt, opts = {}) {
    const s = stimulus(t, events);
    const N = neuralResponse(s, p.kappa, p.tau_I, dt);
    const kLen = Math.min(t.length, Math.ceil(40 / dt));   // kernels are ~0 beyond 40 s
    const h = gammaKernel(kLen, dt, TAU_F), g = gammaKernel(kLen, dt, TAU_M);
    const m1 = 1 + (p.f1 - 1) / p.n;
    const cbf = (deltaT) => convolveDelayed(N, h, DELTA_T_M + deltaT, dt).map((x) => 1 + (p.f1 - 1) * x);
    const m = convolveDelayed(N, g, DELTA_T_M, dt).map((x) => 1 + (m1 - 1) * x);
    const f = cbf(p.delta_t);
    const { q, v } = balloon(f, m, p.tau_MTT, p.tau_plus, p.tau_minus, p.alpha, dt);
    const out = { s, N, f, m, q, v, b: bold(q, v, p.V0), m1 };
    if (opts.variants) {
      out.fig4 = {};
      for (const [key, tm] of Object.entries(FIG4)) {
        const r = balloon(cbf(tm.delta_t), m, p.tau_MTT, tm.tau_plus, tm.tau_minus, p.alpha, dt);
        out.fig4[key] = { b: bold(r.q, r.v, p.V0), q: r.q, v: r.v };
      }
    }
    return out;
  }

  /**
   * Fig. 5 linear prediction: the responses to one 1-s event at the first onset, shifted to every
   * 1-s piece of the stimulus and summed (f − 1 and b). Exact superposition only if the whole
   * chain were linear -- the gap to the true curve is the nonlinearity (neural κ, BOLD ceiling).
   */
  function linearPrediction(t, events, p, dt) {
    const on0 = events[0][0];
    const single = simulate(t, [[on0, 1]], p, dt);
    const n = t.length, fp = new Float64Array(n), bp = new Float64Array(n);
    for (const [on, dur] of events) {
      for (let k = 0; k < Math.round(dur); k++) {
        const shift = Math.round((on + k - on0) / dt);
        for (let i = shift; i < n; i++) {
          fp[i] += single.f[i - shift] - 1;
          bp[i] += single.b[i - shift];
        }
      }
    }
    return { f: fp.map((x) => 1 + x), b: bp };
  }

  const api = { DEFAULTS, FIG4, ONSET, A1, A2, stimulusEvents, timeGrid, simulate, linearPrediction, bold };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BalloonModel = api;
})(typeof window !== "undefined" ? window : globalThis);
