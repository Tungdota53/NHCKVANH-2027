(function () {
  "use strict";

  const OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#334155",
  ];

  function esc(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function finite(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function svgWrap(w, h, inner) {
    return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Biểu đồ khoa học">${inner}</svg>`;
  }

  function emptyState(el, msg) {
    if (!el) return;
    el.innerHTML = `<div class="subtitle" style="padding:1.25rem;text-align:center;">${esc(msg)}</div>`;
  }

  function axisFrame(w, h, pad, xMin, xMax, yMin, yMax, xLabel, yLabel, opts) {
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const xSpan = Math.max(1e-9, xMax - xMin);
    const ySpan = Math.max(1e-9, yMax - yMin);
    const sx = (x) => pad.left + ((x - xMin) / xSpan) * plotW;
    const sy = (y) => pad.top + plotH - ((y - yMin) / ySpan) * plotH;
    const hideXTicks = Boolean(opts && opts.hideXTicks);

    let g = `<rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="none" stroke="var(--border)" stroke-width="1"/>`;
    for (let i = 0; i <= 4; i++) {
      const f = i / 4;
      const gx = pad.left + f * plotW;
      const gy = pad.top + f * plotH;
      const xv = xMin + f * xSpan;
      const yv = yMax - f * ySpan;
      if (!hideXTicks) {
        g += `<line x1="${gx.toFixed(1)}" y1="${pad.top}" x2="${gx.toFixed(1)}" y2="${pad.top + plotH}" stroke="var(--border)" stroke-opacity="0.35" stroke-dasharray="3,3"/>`;
        g += `<text x="${gx.toFixed(1)}" y="${h - pad.bottom + 16}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${xv.toFixed(xv >= 100 ? 0 : 2)}</text>`;
      }
      g += `<line x1="${pad.left}" y1="${gy.toFixed(1)}" x2="${pad.left + plotW}" y2="${gy.toFixed(1)}" stroke="var(--border)" stroke-opacity="0.35" stroke-dasharray="3,3"/>`;
      g += `<text x="${pad.left - 8}" y="${(gy + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="var(--text-muted)">${yv.toFixed(yv >= 100 ? 0 : 2)}</text>`;
    }
    g += `<text x="${pad.left + plotW / 2}" y="${h - 6}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)">${esc(xLabel)}</text>`;
    g += `<text x="14" y="${pad.top + plotH / 2}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)" transform="rotate(-90 14 ${pad.top + plotH / 2})">${esc(yLabel)}</text>`;
    return { g, sx, sy, plotW, plotH };
  }

  function renderParetoChart(el, candidates, decision, detMetric) {
    if (!el) return;
    const pts = (candidates || []).filter((c) => finite(c.quality_mean) && finite(c.time_mean));
    if (!pts.length) return emptyState(el, "Chưa có ứng viên hoàn chỉnh để vẽ biên Pareto.");
    const w = 640, h = 320;
    const pad = { top: 20, right: 24, bottom: 44, left: 58 };
    const xs = pts.map((p) => p.time_mean);
    const ys = pts.map((p) => p.quality_mean);
    const ms = pts.map((p) => (finite(p.memory_max) ? p.memory_max : 512));
    const xMin = Math.max(0, Math.min(...xs) * 0.88);
    const xMax = Math.max(...xs) * 1.08 + 1e-6;
    const yMin = Math.max(0, Math.min(...ys) - 0.05);
    const yMax = Math.min(1.02, Math.max(...ys) + 0.04);
    const mMin = Math.min(...ms), mMax = Math.max(...ms);
    const yLabel = detMetric === "map50_95" ? "Val mAP50-95" : detMetric === "map50" ? "Val mAP50" : "Chất lượng Val";
    const { g, sx, sy } = axisFrame(w, h, pad, xMin, xMax, yMin, yMax, "Thời gian huấn luyện (giây) — thấp là nhanh hơn", yLabel + " — cao là tốt hơn");

    const paretoIds = new Set((decision && decision.pareto) || []);
    const selIdx = decision ? decision.selected_index : null;
    const paretoPts = pts.filter((p) => paretoIds.has(p.config_id)).sort((a, b) => a.time_mean - b.time_mean);
    let inner = g;
    if (paretoPts.length > 1) {
      const pathD = paretoPts.map((p, idx) => `${idx === 0 ? "M" : "L"}${sx(p.time_mean).toFixed(1)},${sy(p.quality_mean).toFixed(1)}`).join(" ");
      inner += `<path d="${pathD}" fill="none" stroke="${OKABE_ITO[2]}" stroke-width="2" stroke-dasharray="5,3"/>`;
    }
    pts.forEach((p) => {
      const cx = sx(p.time_mean), cy = sy(p.quality_mean);
      const mem = finite(p.memory_max) ? p.memory_max : mMin;
      const r = 5 + (mMax > mMin ? ((mem - mMin) / (mMax - mMin)) * 7 : 2);
      const isSel = p.index === selIdx;
      const isPar = paretoIds.has(p.config_id);
      const fill = isSel ? OKABE_ITO[3] : isPar ? OKABE_ITO[2] : OKABE_ITO[0];
      const stroke = isSel ? "#000000" : "var(--bg-elev)";
      const model = (p.config && p.config.model ? String(p.config.model).split(/[\\/]/).pop() : "cfg");
      const memTxt = finite(p.memory_max) ? `${p.memory_max.toFixed(1)} MiB` : "null";
      const tip = `c${String(p.index).padStart(2, "0")} (${model}) | Q=${p.quality_mean.toFixed(4)} | Thời gian=${p.time_mean.toFixed(2)}s | Bộ nhớ=${memTxt}`;
      inner += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="${fill}" fill-opacity="0.86" stroke="${stroke}" stroke-width="${isSel ? 2.5 : 1.5}"><title>${esc(tip)}</title></circle>`;
      if (isSel || isPar) {
        inner += `<text x="${(cx + r + 3).toFixed(1)}" y="${(cy + 4).toFixed(1)}" font-size="11" font-weight="700" fill="${fill}">${isSel ? "★ " : ""}c${String(p.index).padStart(2, "0")}</text>`;
      }
    });
    inner += `<text x="${w - pad.right}" y="${pad.top + 14}" text-anchor="end" font-size="11" fill="var(--text-muted)">Cỡ chấm ~ VRAM/RAM | ★ Đã chọn | Xanh lá = Pareto</text>`;
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderLearningCurvesChart(el, stateResults, selectedIdx, detMetric) {
    if (!el) return;
    const entries = Object.entries(stateResults || {});
    const series = [];
    entries.forEach(([idxStr, runs]) => {
      (runs || []).forEach((r) => {
        if (r && r.mode === "train" && Array.isArray(r.learning_curve) && r.learning_curve.length) {
          series.push({ idx: Number(idxStr), seed: r.seed, curve: r.learning_curve });
        }
      });
    });
    if (!series.length) return emptyState(el, "Chưa ghi nhận đường học theo epoch.");
    const w = 640, h = 320;
    const pad = { top: 20, right: 24, bottom: 44, left: 58 };
    let maxEp = 1, minY = 1, maxY = 0;
    const yKey = detMetric === "map50_95" ? "map50_95" : "map50";
    series.forEach((s) => {
      s.curve.forEach((pt) => {
        const ep = Number(pt.epoch || 0);
        const val = finite(pt[yKey]) ? pt[yKey] : finite(pt.quality_score) ? pt.quality_score : finite(pt.quality) ? pt.quality : null;
        if (ep > maxEp) maxEp = ep;
        if (val !== null) {
          minY = Math.min(minY, val);
          maxY = Math.max(maxY, val);
        }
      });
    });
    if (maxY <= minY) { minY = 0; maxY = 1; }
    const yTitle = detMetric === "map50_95" ? "mAP50-95" : detMetric === "map50" ? "mAP50" : "Chất lượng";
    const { g, sx, sy } = axisFrame(w, h, pad, 1, Math.max(2, maxEp), Math.max(0, minY - 0.05), Math.min(1.02, maxY + 0.05), "Epoch", yTitle);
    let inner = g;
    series.slice(0, 24).forEach((s, i) => {
      const isSel = s.idx === selectedIdx;
      const color = isSel ? OKABE_ITO[3] : OKABE_ITO[i % OKABE_ITO.length];
      const pts = s.curve.map((pt) => {
        const ep = Number(pt.epoch || 1);
        const val = finite(pt[yKey]) ? pt[yKey] : finite(pt.quality_score) ? pt.quality_score : finite(pt.quality) ? pt.quality : 0;
        return { x: sx(ep), y: sy(val), ep, val };
      });
      const coords = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`);
      const title = `c${String(s.idx).padStart(2, "0")} seed ${esc(s.seed)}`;
      if (coords.length > 1) {
        inner += `<polyline fill="none" stroke="${color}" stroke-width="${isSel ? 2.6 : 1.3}" stroke-opacity="${isSel ? 1 : 0.55}" points="${coords.join(" ")}"><title>${title}</title></polyline>`;
      }
      if (pts.length === 1 || pts.length <= 30) {
        pts.forEach((p) => {
          inner += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${isSel ? 4 : 3}" fill="${color}" fill-opacity="${isSel ? 1 : 0.75}"><title>${title} | epoch ${esc(p.ep)}: ${p.val.toFixed(4)}</title></circle>`;
        });
      }
    });
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderSurrogateTraceChart(el, acqHistory, candidates) {
    if (!el) return;
    const byIdx = {};
    (candidates || []).forEach((c) => {
      if (finite(c.quality_mean)) byIdx[String(c.index)] = c.quality_mean;
    });
    const lastWithDiag = [...(acqHistory || [])].reverse().find((h) => h && (h.surrogate_diagnostics || h.scores));
    if (!lastWithDiag) return emptyState(el, "Chưa có dữ liệu dự báo Surrogate (thuật toán không dùng surrogate).");
    const preds = (lastWithDiag.surrogate_diagnostics && lastWithDiag.surrogate_diagnostics.predictions) || {};
    const scores = lastWithDiag.scores || {};
    const points = [];
    Object.keys(byIdx).forEach((idx) => {
      const p = preds[idx] || {};
      const sc = scores[idx] || {};
      const mean = finite(p.mean) ? p.mean : finite(sc.quality_mean_prediction) ? sc.quality_mean_prediction : null;
      const std = finite(p.std) ? p.std : finite(sc.uncertainty) ? sc.uncertainty : 0;
      if (mean !== null) points.push({ idx, obs: byIdx[idx], pred: mean, std });
    });
    if (!points.length) return emptyState(el, "Chưa đủ cặp điểm dự đoán vs thực đo.");
    const w = 640, h = 320;
    const pad = { top: 24, right: 24, bottom: 44, left: 58 };
    const vals = points.flatMap((p) => [p.obs, p.pred - p.std, p.pred + p.std]);
    const lo = Math.max(0, Math.min(...vals) - 0.05);
    const hi = Math.min(1.02, Math.max(...vals) + 0.05);
    const { g, sx, sy } = axisFrame(w, h, pad, lo, hi, lo, hi, "Chất lượng thực đo (Validation)", "Surrogate dự đoán (TB ± độ lệch chuẩn)");
    let inner = g;
    inner += `<line x1="${sx(lo).toFixed(1)}" y1="${sy(lo).toFixed(1)}" x2="${sx(hi).toFixed(1)}" y2="${sy(hi).toFixed(1)}" stroke="var(--text-muted)" stroke-dasharray="4,4"/>`;
    points.forEach((p, i) => {
      const cx = sx(p.obs), cy = sy(p.pred);
      const yTop = sy(Math.min(hi, p.pred + p.std)), yBot = sy(Math.max(lo, p.pred - p.std));
      const col = OKABE_ITO[i % OKABE_ITO.length];
      inner += `<line x1="${cx.toFixed(1)}" y1="${yTop.toFixed(1)}" x2="${cx.toFixed(1)}" y2="${yBot.toFixed(1)}" stroke="${col}" stroke-width="1.5"/>`;
      inner += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5" fill="${col}"><title>c${esc(p.idx)}: thực đo=${p.obs.toFixed(4)}, dự đoán=${p.pred.toFixed(4)}±${p.std.toFixed(4)}</title></circle>`;
    });
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderAnytimeChart(el, anytime) {
    if (!el) return;
    const traj = (anytime && anytime.trajectory) || [];
    if (!traj.length) return emptyState(el, "Chưa có dữ liệu hội tụ theo thời gian thực.");
    const w = 640, h = 300;
    const pad = { top: 20, right: 24, bottom: 44, left: 58 };
    const times = traj.map((p) => Number(p.active_seconds ?? p.t_seconds ?? p.time_seconds ?? 0));
    const utils = traj.map((p) => Number(p.best_utility ?? p.utility ?? 0));
    const xMax = Math.max(1, ...times);
    const uMin = Math.min(...utils);
    const uMax = Math.max(...utils);
    const span = Math.max(0.05, (uMax - uMin) * 1.2);
    const yMin = uMin - 0.1 * span;
    const yMax = yMin + span;
    const { g, sx, sy } = axisFrame(w, h, pad, 0, xMax, yMin, yMax, "Thời gian tìm kiếm (giây)", "Điểm hữu ích U tốt nhất");
    let inner = g;
    const coords = traj.map((p, i) => `${sx(times[i]).toFixed(1)},${sy(utils[i]).toFixed(1)}`);
    inner += `<polyline fill="none" stroke="${OKABE_ITO[0]}" stroke-width="2.5" points="${coords.join(" ")}"/>`;
    traj.forEach((p, i) => {
      inner += `<circle cx="${sx(times[i]).toFixed(1)}" cy="${sy(utils[i]).toFixed(1)}" r="4" fill="${OKABE_ITO[0]}"><title>t=${times[i].toFixed(1)}s | U=${utils[i].toFixed(4)}</title></circle>`;
    });
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderPerClassChart(el, perClass) {
    if (!el) return;
    const rows = Array.isArray(perClass) ? perClass : [];
    if (!rows.length) return emptyState(el, "Chưa có dữ liệu AP từng lớp.");
    const w = 640, h = 300;
    const pad = { top: 20, right: 20, bottom: 54, left: 54 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    let inner = `<rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="none" stroke="var(--border)"/>`;
    const slot = plotW / rows.length;
    const barW = Math.max(10, Math.min(42, slot * 0.36));
    rows.forEach((r, i) => {
      const ap50 = finite(r.ap50) ? r.ap50 : finite(r.ap50_mean) ? r.ap50_mean : null;
      const ap95 = finite(r.ap50_95) ? r.ap50_95 : finite(r.ap50_95_mean) ? r.ap50_95_mean : null;
      const cx = pad.left + slot * (i + 0.5);
      if (ap50 !== null) {
        const h50 = ap50 * plotH;
        inner += `<rect x="${(cx - barW - 1).toFixed(1)}" y="${(pad.top + plotH - h50).toFixed(1)}" width="${barW.toFixed(1)}" height="${h50.toFixed(1)}" fill="${OKABE_ITO[0]}"><title>${esc(r.class_name)}: AP50=${ap50.toFixed(4)}</title></rect>`;
      }
      if (ap95 !== null) {
        const h95 = ap95 * plotH;
        inner += `<rect x="${(cx + 1).toFixed(1)}" y="${(pad.top + plotH - h95).toFixed(1)}" width="${barW.toFixed(1)}" height="${h95.toFixed(1)}" fill="${OKABE_ITO[1]}"><title>${esc(r.class_name)}: AP50-95=${ap95.toFixed(4)}</title></rect>`;
      }
      if (ap50 === null && ap95 === null) {
        inner += `<text x="${cx.toFixed(1)}" y="${(pad.top + plotH - 8).toFixed(1)}" text-anchor="middle" font-size="10" fill="var(--text-muted)"><title>${esc(r.class_name)}: null</title>null</text>`;
      }
      inner += `<text x="${cx.toFixed(1)}" y="${h - 24}" text-anchor="middle" font-size="11" fill="var(--text)">${esc(r.class_name || r.class_id)}</text>`;
    });
    inner += `<text x="${w - pad.right}" y="${pad.top + 14}" text-anchor="end" font-size="11" fill="var(--text-muted)">Xanh dương = AP@50 | Cam = AP@50-95</text>`;
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderCurveChart(el, xGrid, meanY, perClass, xLabel, yLabel) {
    if (!el) return;
    if (!Array.isArray(xGrid) || !Array.isArray(meanY) || !xGrid.length) {
      return emptyState(el, "Chưa có dữ liệu đường cong.");
    }
    const w = 520, h = 280;
    const pad = { top: 20, right: 20, bottom: 44, left: 52 };
    const { g, sx, sy } = axisFrame(w, h, pad, 0, 1, 0, 1, xLabel, yLabel);
    let inner = g;
    if (Array.isArray(perClass)) {
      perClass.slice(0, 8).forEach((cls, idx) => {
        const ys = cls.precision || cls.f1 || [];
        if (ys.length === xGrid.length) {
          const pts = xGrid
            .map((x, i) => (finite(x) && finite(ys[i]) ? `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}` : null))
            .filter(Boolean)
            .join(" ");
          if (pts) {
            inner += `<polyline fill="none" stroke="${OKABE_ITO[(idx + 1) % OKABE_ITO.length]}" stroke-width="1.2" stroke-opacity="0.55" points="${pts}"><title>${esc(cls.class_name || idx)}</title></polyline>`;
          }
        }
      });
    }
    const meanPts = xGrid
      .map((x, i) => (finite(x) && finite(meanY[i]) ? `${sx(x).toFixed(1)},${sy(meanY[i]).toFixed(1)}` : null))
      .filter(Boolean)
      .join(" ");
    if (meanPts) {
      inner += `<polyline fill="none" stroke="${OKABE_ITO[0]}" stroke-width="2.6" points="${meanPts}"/>`;
    }
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderConfusionMatrix(el, cm) {
    if (!el) return;
    const mat = cm && cm.normalized;
    const labels = (cm && cm.labels) || [];
    if (!Array.isArray(mat) || !mat.length) return emptyState(el, "Chưa có dữ liệu ma trận nhầm lẫn.");
    const n = mat.length;
    const cell = Math.max(36, Math.min(64, Math.floor(320 / n)));
    const left = 110, top = 26;
    const w = left + n * cell + 24;
    const h = top + n * cell + 64;
    let inner = "";
    for (let r = 0; r < n; r++) {
      const rLab = labels[r] !== undefined ? labels[r] : `c${r}`;
      inner += `<text x="${left - 8}" y="${top + r * cell + cell / 2 + 4}" text-anchor="end" font-size="11" fill="var(--text)">${esc(rLab)}</text>`;
      for (let c = 0; c < n; c++) {
        const v = Number((mat[r] && mat[r][c]) || 0);
        const alpha = Math.min(0.95, Math.max(0.06, v));
        const x = left + c * cell, y = top + r * cell;
        inner += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="#0072B2" fill-opacity="${alpha.toFixed(2)}" stroke="var(--bg-elev)"/>`;
        inner += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" font-size="10" font-weight="600" fill="${v > 0.5 ? "#ffffff" : "var(--text)"}">${(v * 100).toFixed(0)}%</text>`;
      }
    }
    for (let c = 0; c < n; c++) {
      const cLab = labels[c] !== undefined ? labels[c] : `c${c}`;
      inner += `<text x="${left + c * cell + cell / 2}" y="${top + n * cell + 16}" text-anchor="middle" font-size="11" fill="var(--text)">${esc(cLab)}</text>`;
    }
    el.innerHTML = svgWrap(w, h, inner);
  }

  function renderBenchmarkChart(el, summaries) {
    if (!el) return;
    const rows = Array.isArray(summaries) ? summaries : [];
    if (!rows.length) return emptyState(el, "Chưa có dữ liệu tổng hợp đối chứng.");
    const w = 640, h = 300;
    const pad = { top: 24, right: 24, bottom: 52, left: 60 };
    const finiteVals = rows.map((r) => (r ? r.conditional_regret_mean : null)).filter(finite);
    const yMax = Math.max(0.05, ...finiteVals) * 1.15;
    const { g, plotW, plotH, sy } = axisFrame(w, h, pad, 0, rows.length, 0, yMax, "Thuật toán", "TB Objective Regret (thấp hơn là tốt hơn)", { hideXTicks: true });
    let inner = g;
    const slot = plotW / rows.length;
    const bw = Math.max(18, Math.min(54, slot * 0.5));
    const baseY = pad.top + plotH;
    rows.forEach((r, i) => {
      const cx = pad.left + slot * (i + 0.5);
      const methodLabel = r ? r.method : "";
      if (r && finite(r.conditional_regret_mean)) {
        const v = r.conditional_regret_mean;
        const topY = sy(v);
        const bh = Math.max(0, baseY - topY);
        inner += `<rect x="${(cx - bw / 2).toFixed(1)}" y="${topY.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="${OKABE_ITO[i % OKABE_ITO.length]}"><title>${esc(methodLabel)}: regret=${v.toFixed(6)}</title></rect>`;
      } else {
        inner += `<g class="null-regret"><title>${esc(methodLabel)}: không có cấu hình khả thi xác nhận (regret=null)</title><line x1="${(cx - bw / 2).toFixed(1)}" y1="${baseY.toFixed(1)}" x2="${(cx + bw / 2).toFixed(1)}" y2="${baseY.toFixed(1)}" stroke="var(--text-muted)" stroke-width="2" stroke-dasharray="4,3"/><text x="${cx.toFixed(1)}" y="${(baseY - 8).toFixed(1)}" text-anchor="middle" font-size="11" font-weight="600" fill="var(--text-muted)">null</text></g>`;
      }
      inner += `<text x="${cx.toFixed(1)}" y="${h - 24}" text-anchor="middle" font-size="11" fill="var(--text)">${esc(methodLabel)}</text>`;
    });
    el.innerHTML = svgWrap(w, h, inner);
  }

  window.NCKHCharts = {
    renderParetoChart,
    renderLearningCurvesChart,
    renderSurrogateTraceChart,
    renderAnytimeChart,
    renderPerClassChart,
    renderCurveChart,
    renderConfusionMatrix,
    renderBenchmarkChart,
  };
})();
