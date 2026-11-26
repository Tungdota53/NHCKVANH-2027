(function () {
  "use strict";

  const THEME_KEY = "nckh_ui_theme";

  function applyTheme(theme) {
    const t = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch (_) {}
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = t === "dark" ? "Giao diện sáng" : "Giao diện tối";
  }

  function initTheme() {
    let saved = "light";
    try {
      saved = localStorage.getItem(THEME_KEY) || "light";
    } catch (_) {}
    applyTheme(saved);
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const cur = document.documentElement.getAttribute("data-theme");
        applyTheme(cur === "dark" ? "light" : "dark");
      });
    }
  }

  function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  async function apiGet(url) {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken(),
      },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function esc(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(v, digits = 4) {
    return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
  }

  function metricScalar(v) {
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (v && typeof v === "object" && typeof v.mean === "number" && Number.isFinite(v.mean)) return v.mean;
    return null;
  }

  function baseName(p) {
    if (!p) return "—";
    const s = String(p).split(/[\\/]/);
    return s[s.length - 1] || String(p);
  }

  function badgeClass(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("no_verified") || s.includes("timeout") || s.includes("partial") || s.includes("interrupted") || s === "unresolved" || s.includes("modified_protocol")) {
      return "badge-warn";
    }
    if (s.includes("fail") || s.includes("oom") || s.includes("contended") || s.includes("infeasible") || s === "error") {
      return "badge-danger";
    }
    if (s.includes("optimal") || s === "complete" || s === "confirmed_feasible" || s === "ok") {
      return "badge-success";
    }
    if (s.includes("best") || s.includes("fully_trained") || s === "search" || s === "prepared" || s === "running") {
      return "badge-info";
    }
    return "badge-muted";
  }

  function statusLabel(status) {
    const raw = String(status || "");
    const s = raw.toLowerCase();
    if (s === "empirical_optimal_in_declared_space" || s === "empirical_optimal") return "Tối ưu thực nghiệm";
    if (s === "fully_trained_best_observed" || s === "observed_best_in_explored_space" || s === "observed_best") return "Tốt nhất đã đo";
    if (s === "contended_smoke_not_certified") return "Tranh tài nguyên (Không chứng nhận)";
    if (s === "modified_protocol_optimality_not_certified") return "Đã sửa protocol";
    if (s === "confirmed_feasible") return "Đạt khả thi";
    if (s === "ok") return "Đạt";
    if (s === "complete") return "Hoàn tất";
    if (s === "search") return "Đang tìm kiếm";
    if (s === "running") return "Đang chạy";
    if (s === "prepared") return "Đã lập kế hoạch";
    if (s === "pending") return "Chờ chạy";
    if (s === "timeout") return "Hết giờ (Timeout)";
    if (s === "unresolved") return "Chưa đủ seed";
    if (s === "no_verified_feasible_selection" || s === "no_verified_feasible") return "Chưa có cấu hình khả thi";
    if (s === "observed_required_seed_oom" || s === "oom") return "Tràn VRAM (OOM)";
    if (s === "confirmed_constraint_failure") return "Vi phạm ràng buộc";
    if (s === "infeasible") return "Không khả thi";
    if (s === "interrupted") return "Đã tạm dừng";
    if (s === "error" || s === "failed") return "Lỗi chạy";
    return raw || "—";
  }

  function kindLabel(kind) {
    if (kind === "v2") return "Tìm kiếm v2";
    if (kind === "v2_deployment") return "Triển khai v2";
    if (kind === "research") return "Nghiên cứu v1";
    if (kind === "legacy_training") return "Huấn luyện cũ";
    if (kind === "benchmark") return "Đối chứng Benchmark";
    if (kind === "algorithm_study") return "Nghiên cứu đa bộ";
    return kind || "—";
  }

  function queryParam(name) {
    return new URLSearchParams(window.location.search).get(name);
  }

  // ---- View 1: Tổng quan & Phần cứng ----
  async function initOverview() {
    const hwEl = document.getElementById("hw-cards");
    const sumEl = document.getElementById("overview-summary");
    const runsBody = document.getElementById("runs-table-body");
    const bmBody = document.getElementById("benchmarks-table-body");
    const filterInput = document.getElementById("runs-filter");
    const kindSelect = document.getElementById("runs-kind-filter");
    let cachedRuns = [];

    async function refreshHw() {
      try {
        const hw = await apiGet("/api/hardware");
        const gpu = hw.gpus && hw.gpus[0];
        const vramPct = gpu && gpu.vram_total_mb ? Math.min(100, Math.round((gpu.vram_used_mb / gpu.vram_total_mb) * 100)) : 0;
        const cpuPct = Math.min(100, Math.round(hw.cpu_percent || 0));
        const ramPct = Math.min(100, Math.round(hw.ram_percent || 0));
        const diskUsed = Math.max(0, (hw.disk_total_mb || 0) - (hw.disk_free_mb || 0));
        const diskPct = hw.disk_total_mb ? Math.min(100, Math.round((diskUsed / hw.disk_total_mb) * 100)) : 0;
        const gpuCard = gpu
          ? `<div class="card"><div class="kpi-label">Card đồ họa (GPU)</div><div class="kpi-value" style="font-size:1.05rem;">${esc(gpu.name)}</div><div class="kpi-sub">VRAM: ${fmt(gpu.vram_used_mb, 0)} / ${fmt(gpu.vram_total_mb, 0)} MiB · Nhiệt: ${gpu.temperature_c !== null ? fmt(gpu.temperature_c, 0) + "°C" : "N/A"} · Tải: ${gpu.gpu_utilization_percent !== null ? fmt(gpu.gpu_utilization_percent, 0) + "%" : "N/A"}</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${vramPct}%"></div></div></div>`
          : `<div class="card"><div class="kpi-label">Chế độ tính toán</div><div class="kpi-value">CPU</div><div class="kpi-sub">Không có CUDA; torch ${esc(hw.torch_version || "N/A")}</div></div>`;
        hwEl.innerHTML = `
          ${gpuCard}
          <div class="card"><div class="kpi-label">Vi xử lý (CPU)</div><div class="kpi-value">${fmt(hw.cpu_percent, 1)}%</div><div class="kpi-sub">${hw.logical_cpus} luồng xử lý · Chế độ ${esc(hw.mode.toUpperCase())}</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${cpuPct}%"></div></div></div>
          <div class="card"><div class="kpi-label">Bộ nhớ hệ thống (RAM)</div><div class="kpi-value">${fmt(hw.ram_used_mb, 0)} MiB</div><div class="kpi-sub">Còn trống: ${fmt(hw.ram_available_mb, 0)} / ${fmt(hw.ram_total_mb, 0)} MiB (${fmt(hw.ram_percent, 1)}%)</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${ramPct}%"></div></div></div>
          <div class="card"><div class="kpi-label">Dung lượng ổ đĩa trống</div><div class="kpi-value">${fmt(hw.disk_free_mb / 1024, 1)} GB</div><div class="kpi-sub">Đã dùng ${diskPct}% / Tổng ${fmt(hw.disk_total_mb / 1024, 1)} GB</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${diskPct}%"></div></div></div>
        `;
      } catch (err) {
        if (hwEl) hwEl.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
      }
    }

    function renderRuns() {
      const q = (filterInput && filterInput.value || "").trim().toLowerCase();
      const k = (kindSelect && kindSelect.value) || "all";
      const filtered = cachedRuns.filter((r) => {
        if (k !== "all" && r.kind !== k) return false;
        if (!q) return true;
        return [r.rel_path, r.task, r.method, r.verification_status, r.selected_config_id].some((x) => String(x || "").toLowerCase().includes(q));
      });
      if (!filtered.length) {
        runsBody.innerHTML = `<tr><td colspan="9" class="subtitle">Chưa có lượt chạy phù hợp trong thư mục kết quả.</td></tr>`;
        return;
      }
      runsBody.innerHTML = filtered.map((r) => {
        const rel = String(r.rel_path || r.name || "");
        const parts = rel.split("/").filter(Boolean);
        const leaf = parts.pop() || rel;
        const parent = parts.join("/");
        const st = r.verification_status || r.phase;
        return `
        <tr>
          <td title="${esc(rel)}">
            <div class="run-name-main">${esc(leaf)}</div>
            ${parent ? `<div class="run-name-sub">${esc(parent)}</div>` : ""}
          </td>
          <td>${esc(kindLabel(r.kind))}</td>
          <td>${esc(r.task)}</td>
          <td>${esc(r.method)}</td>
          <td><span class="badge ${badgeClass(st)}" title="${esc(st)}">${esc(statusLabel(st))}</span></td>
          <td class="mono">${esc(r.selected_config_id || "—")}</td>
          <td class="num">${fmt(r.best_quality, 4)}</td>
          <td class="num">${fmt(r.utility, 4)}</td>
          <td>
            <a class="btn btn-sm" href="/analysis?run=${encodeURIComponent(rel)}">Phân tích</a>
            <a class="btn btn-sm" href="/monitor?run=${encodeURIComponent(rel)}">Theo dõi</a>
            <a class="btn btn-sm" href="/export?run=${encodeURIComponent(rel)}">Xuất</a>
          </td>
        </tr>
      `;
      }).join("");
    }

    async function loadRuns() {
      try {
        const data = await apiGet("/api/runs");
        cachedRuns = data.runs || [];
        const bms = [...(data.benchmarks || []), ...(data.algorithm_studies || [])];
        if (sumEl) {
          const totalRuns = cachedRuns.length;
          const okCount = cachedRuns.filter((r) => {
            const st = String(r.verification_status || "");
            return st.includes("optimal") || st.includes("best") || st.includes("fully_trained");
          }).length;
          const evalCount = cachedRuns.filter((r) => r.has_evaluation).length;
          sumEl.innerHTML = `
            <div class="card"><div class="kpi-label">Tổng số lượt chạy</div><div class="kpi-value">${totalRuns}</div><div class="kpi-sub">Đã phát hiện trong thư mục kết quả</div></div>
            <div class="card"><div class="kpi-label">Chọn được cấu hình</div><div class="kpi-value">${okCount}</div><div class="kpi-sub">Đạt tối ưu / tốt nhất trong không gian</div></div>
            <div class="card"><div class="kpi-label">Đã đánh giá tập Test</div><div class="kpi-value">${evalCount}</div><div class="kpi-sub">Có đầy đủ Latency, PR, F1, Confusion Matrix</div></div>
            <div class="card"><div class="kpi-label">Bộ đối chứng thuật toán</div><div class="kpi-value">${bms.length}</div><div class="kpi-sub">Benchmark &amp; nghiên cứu đa bộ dữ liệu</div></div>
          `;
        }
        renderRuns();
        if (bmBody) {
          bmBody.innerHTML = bms.length
            ? bms.map((b) => `
              <tr>
                <td class="mono">${esc(b.rel_path || b.name)}</td>
                <td>${esc(kindLabel(b.kind))}</td>
                <td>${esc(b.task || "Đa nghiên cứu")}</td>
                <td>${esc((b.methods || []).join(", ") || (b.study_count ? b.study_count + " nghiên cứu" : "—"))}</td>
                <td><a class="btn btn-sm" href="/benchmark?path=${encodeURIComponent(b.rel_path)}">Mở báo cáo</a></td>
              </tr>
            `).join("")
            : `<tr><td colspan="5" class="subtitle">Chưa phát hiện báo cáo đối chứng hoặc algorithm study nào.</td></tr>`;
        }
      } catch (err) {
        if (runsBody) runsBody.innerHTML = `<tr><td colspan="9">${esc(err.message)}</td></tr>`;
      }
    }

    if (filterInput) filterInput.addEventListener("input", renderRuns);
    if (kindSelect) kindSelect.addEventListener("change", renderRuns);
    const refreshBtn = document.getElementById("refresh-hw-btn");
    if (refreshBtn) refreshBtn.addEventListener("click", refreshHw);
    await Promise.all([refreshHw(), loadRuns()]);
    setInterval(refreshHw, 3000);
  }

  // ---- View 2: Theo dõi trực tiếp ----
  async function initMonitor() {
    const sel = document.getElementById("monitor-run-select");
    const refreshBtn = document.getElementById("monitor-refresh-btn");
    const stopBtn = document.getElementById("safe-stop-btn");
    const msgEl = document.getElementById("monitor-status-msg");
    const diagEl = document.getElementById("monitor-diagnostic-banner");
    const kpiEl = document.getElementById("monitor-kpis");
    const candBody = document.getElementById("monitor-candidates-body");
    const logTitleEl = document.getElementById("monitor-log-title");
    const logEl = document.getElementById("monitor-worker-log");
    const jobsBody = document.getElementById("monitor-jobs-body");

    let pollCount = 0;

    async function refreshRunList(preferRun) {
      const runsData = await apiGet("/api/runs");
      const runs = runsData.runs || [];
      const cur = preferRun !== undefined ? preferRun : sel.value;
      sel.innerHTML = `<option value="">-- Chọn lượt chạy cần theo dõi --</option>` + runs.map((r) => `<option value="${esc(r.rel_path)}">${esc(r.rel_path)} (${esc(statusLabel(r.verification_status || r.phase))})</option>`).join("");
      if (cur && runs.some((r) => r.rel_path === cur)) {
        sel.value = cur;
      } else if (!cur && runs[0]) {
        sel.value = runs[0].rel_path;
      }
    }

    await refreshRunList(queryParam("run") || undefined);

    async function poll() {
      const target = sel.value;
      const url = target ? `/api/live?run=${encodeURIComponent(target)}` : "/api/live";
      try {
        const data = await apiGet(url);
        const jobs = data.jobs || [];
        if (jobs.some((j) => j.alive) && pollCount % 3 === 2) {
          refreshRunList().catch(() => {});
        }
        pollCount += 1;

        jobsBody.innerHTML = jobs.length
          ? jobs.map((j) => {
            const tail = (j.returncode && j.returncode !== 0 && !j.alive && Array.isArray(j.tail_output) && j.tail_output.length)
              ? `<div class="subtitle" style="margin-top:0.25rem;color:var(--danger);white-space:normal;">${esc(j.tail_output.slice(-2).join(" | "))}</div>`
              : "";
            const stateBadge = j.alive
              ? `<span class="badge badge-info">ĐANG CHẠY</span>`
              : `<span class="badge ${j.returncode === 0 ? "badge-muted" : "badge-danger"}">${j.returncode === 0 ? "HOÀN TẤT (0)" : "LỖI (" + j.returncode + ")"}</span>`;
            const stopCell = j.alive
              ? `<button type="button" class="btn btn-sm btn-danger stop-job-btn" data-job-id="${esc(j.job_id)}">Dừng</button>`
              : "—";
            return `<tr><td class="mono">${esc(j.job_id)}</td><td>${esc(j.mode)}</td><td class="mono">${esc(j.pid)}</td><td>${stateBadge}${tail}</td><td class="mono">${esc(j.output)}</td><td>${stopCell}</td></tr>`;
          }).join("")
          : `<tr><td colspan="6" class="subtitle">Chưa có tiến trình nào được khởi chạy từ Web UI trong phiên này.</td></tr>`;

        jobsBody.querySelectorAll(".stop-job-btn").forEach((btn) => {
          btn.addEventListener("click", async () => {
            const jid = btn.getAttribute("data-job-id");
            try {
              const res = await apiPost("/api/stop", { job_id: jid });
              msgEl.innerHTML = `<div class="alert alert-info">Đã gửi lệnh dừng an toàn tới job ${esc(jid)} (${esc(res.status)}).</div>`;
              await poll();
            } catch (err) {
              msgEl.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
            }
          });
        });

        const r = data.run;
        if (!r) {
          if (diagEl) diagEl.innerHTML = "";
          kpiEl.innerHTML = "";
          candBody.innerHTML = "";
          if (logTitleEl) logTitleEl.textContent = "Nhật ký huấn luyện mới nhất (120 dòng cuối)";
          logEl.textContent = "Hãy chọn một lượt chạy ở phía trên để theo dõi tiến độ epoch và nhật ký worker.log.";
          return;
        }

        if (diagEl) {
          if (r.verification_status === "no_verified_feasible_selection" && (r.timeout_count || 0) > 0) {
            diagEl.innerHTML = `
              <div class="alert alert-warn">
                <strong>Lượt chạy này chưa chọn được cấu hình do hết ngân sách thời gian (${r.timeout_count} lần thử bị timeout trong ${fmt(r.budget_seconds, 0)}s).</strong><br/>
                Ngân sách chia cho từng lần thử chưa đủ hoàn thành trọn vẹn 1 cấu hình đủ epoch &amp; seed. Gợi ý: vào <a href="/launcher">Chạy mới</a> và chọn kịch bản <strong>Thử nhanh</strong> (3 epochs · 1 seed · compact) hoặc tăng ngân sách thời gian cho bộ dữ liệu lớn.
              </div>
            `;
          } else if (r.verification_status === "no_verified_feasible_selection") {
            diagEl.innerHTML = `
              <div class="alert alert-warn">
                <strong>Chưa có cấu hình nào hoàn tất đủ điều kiện khả thi</strong> (trạng thái: ${esc(statusLabel(r.verification_status))}). Hãy kiểm tra log bên dưới để xem chi tiết.
              </div>
            `;
          } else {
            diagEl.innerHTML = "";
          }
        }

        const lt = r.latest_trial || {};
        const epPct = lt.trial_epochs ? Math.min(100, Math.round(((lt.epochs_completed || 0) / lt.trial_epochs) * 100)) : 0;
        const phaseStatus = r.verification_status === "no_verified_feasible_selection" ? r.verification_status : r.phase;
        kpiEl.innerHTML = `
          <div class="card"><div class="kpi-label">Trạng thái</div><div class="kpi-value" style="font-size:1.05rem;"><span class="badge ${badgeClass(phaseStatus)}">${esc(statusLabel(phaseStatus))}</span></div><div class="kpi-sub">${esc(r.task)} · ${esc(r.method)} · Hoàn tất ${r.completed_candidates ?? 0}/${r.total_candidates ?? 0}</div></div>
          <div class="card"><div class="kpi-label">Ngân sách còn lại</div><div class="kpi-value">${r.remaining_seconds !== null ? fmt(r.remaining_seconds, 1) + "s" : "—"}</div><div class="kpi-sub">Tổng: ${fmt(r.budget_seconds, 0)}s · Đã chạy: ${fmt(r.elapsed_seconds, 1)}s</div></div>
          <div class="card"><div class="kpi-label">Lần thử mới nhất</div><div class="kpi-value" style="font-size:1.05rem;">${esc(lt.attempt_id || "—")}</div><div class="kpi-sub">Epoch ${lt.epochs_completed ?? 0}/${lt.trial_epochs ?? "?"} · Điểm=${fmt(lt.quality, 4)} · VRAM=${fmt(lt.memory_mb, 1)} MiB</div><div class="progress-track" style="margin-top:0.4rem;"><div class="progress-fill" style="width:${epPct}%"></div></div></div>
          <div class="card"><div class="kpi-label">Tiến trình đang chạy</div><div class="kpi-value">${(r.live_workers || []).length}</div><div class="kpi-sub">${(r.live_workers || []).map((w) => "PID " + w.pid).join(", ") || "Không có tiến trình con nào đang hoạt động"}</div></div>
        `;
        candBody.innerHTML = (r.candidates || []).map((c) => `
          <tr>
            <td class="mono">c${String(c.index).padStart(2, "0")}</td>
            <td>${esc(c.model)}</td>
            <td class="num">${esc(c.imgsz)} / ${esc(c.batch)}</td>
            <td class="num">${esc(c.lr0)}</td>
            <td class="num">${c.attempts}</td>
            <td><span class="badge ${badgeClass(c.status)}" title="${esc(c.status)}">${esc(statusLabel(c.status))}</span></td>
            <td class="num">${fmt(c.best_quality, 4)}</td>
          </tr>
        `).join("");
        if (logTitleEl) {
          logTitleEl.textContent = r.worker_log_attempt
            ? `Nhật ký huấn luyện mới nhất — ${r.worker_log_attempt} (120 dòng cuối)`
            : "Nhật ký huấn luyện mới nhất (120 dòng cuối)";
        }
        logEl.textContent = (r.worker_log && r.worker_log.length) ? r.worker_log.join("\n") : "(Chưa có đầu ra trong worker.log)";
      } catch (err) {
        msgEl.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
      }
    }

    sel.addEventListener("change", poll);
    if (refreshBtn) {
      refreshBtn.addEventListener("click", async () => {
        await refreshRunList();
        await poll();
      });
    }
    if (stopBtn) {
      stopBtn.addEventListener("click", async () => {
        try {
          const res = await apiPost("/api/stop", { run: sel.value || null });
          msgEl.innerHTML = `<div class="alert alert-info">Dừng an toàn: ${esc(res.status)} ${res.pids ? "(PID: " + res.pids.join(", ") + ")" : ""}</div>`;
          await poll();
        } catch (err) {
          msgEl.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
        }
      });
    }
    await poll();
    setInterval(poll, 2500);
  }

  // ---- View 3: Tạo & Chạy thực nghiệm ----
  async function initLauncher() {
    const resData = await apiGet("/api/resources");
    const dsSel = document.getElementById("launch-dataset-select");
    const dataInput = document.getElementById("launch-data");
    const modelSel = document.getElementById("launch-models");
    const methodSel = document.getElementById("launch-method");
    const ruleSel = document.getElementById("launch-rule");
    const cardSel = document.getElementById("launch-conformal-card");
    const tolInput = document.getElementById("launch-quality-tolerance");
    const maxMemInput = document.getElementById("launch-max-memory");
    const spaceSel = document.getElementById("launch-space");
    const epochsInput = document.getElementById("launch-epochs");
    const budgetInput = document.getElementById("launch-budget");
    const seedsInput = document.getElementById("launch-seeds");
    const advisoryEl = document.getElementById("budget-advisory");
    const inspectBtn = document.getElementById("btn-inspect-dataset");
    const inspectOut = document.getElementById("inspect-result");
    const previewEl = document.getElementById("cli-preview");
    const launchBtn = document.getElementById("btn-launch-job");
    const launchMsg = document.getElementById("launch-result");

    const allWeights = resData.weights || [];
    const allDatasets = resData.datasets || [];
    let inspectedTrainCount = null;

    function isLikelyDetectionData() {
      const p = (dataInput.value || "").trim().toLowerCase();
      if (!p) return true;
      if (p.endsWith(".yaml") || p.endsWith(".yml")) return true;
      const matched = allDatasets.find((d) => d.path.toLowerCase() === p);
      return matched ? matched.kind === "detection/yaml" : false;
    }

    function currentApproxImages() {
      if (inspectedTrainCount !== null) return inspectedTrainCount;
      const p = (dataInput.value || "").trim().toLowerCase();
      const matched = allDatasets.find((d) => d.path.toLowerCase() === p);
      return matched && typeof matched.approx_train_images === "number" ? matched.approx_train_images : null;
    }

    if (dsSel && allDatasets.length) {
      dsSel.innerHTML = `<option value="">-- Chọn nhanh bộ dữ liệu có sẵn --</option>` + allDatasets.map((d) => {
        const cntTxt = d.approx_train_images ? ` · ~${d.approx_train_images} ảnh` : "";
        return `<option value="${esc(d.path)}">${esc(d.name)} (${esc(d.kind)}${cntTxt})</option>`;
      }).join("");
      dsSel.addEventListener("change", () => {
        if (dsSel.value) {
          dataInput.value = dsSel.value;
          inspectedTrainCount = null;
          if (!isLikelyDetectionData() && modelSel) modelSel.value = "";
        }
        updatePreview();
      });
      if (!dataInput.value) {
        dataInput.value = allDatasets[0].path;
        dsSel.value = allDatasets[0].path;
      }
    }

    if (modelSel) {
      let opts = "";
      if (allWeights.length > 1) {
        opts += `<option value="__ALL_LOCAL_PT__" selected>Tất cả ${allWeights.length} trọng số .pt trong weights/ (Khuyên dùng cho Detection)</option>`;
      }
      opts += allWeights.map((w, idx) => `<option value="${esc(w.path)}" ${allWeights.length === 1 && idx === 0 ? "selected" : ""}>${esc(w.name)} — ${w.parameters ? (w.parameters / 1e6).toFixed(2) + "M tham số" : w.size_mb + " MB"}${w.gflops_640 ? " | " + w.gflops_640.toFixed(2) + " GFLOPs@640" : ""}</option>`).join("");
      opts += `<option value="">(Không dùng .pt — train .yaml từ đầu hoặc bài toán Classification)</option>`;
      modelSel.innerHTML = opts;
      if (!isLikelyDetectionData()) modelSel.value = "";
    }

    const METHOD_LABELS = {
      conformal_mf: "conformal_mf (Split Conformal VRAM + Multi-Fidelity · Risk-Gated)",
      mf_no_conformal: "mf_no_conformal (Multi-Fidelity không lọc Conformal)",
      tpe_bo: "tpe_bo (Bayesian Optimization TPE · KDE Parzen)",
      adaptive_full: "adaptive_full (GP Kernel Ridge LOOCV · Full-data)",
      adaptive_ehvi: "adaptive_ehvi (3D Pareto Hypervolume EHVI)",
      adaptive_staged_seeds: "adaptive_staged_seeds (Đo tuần tự seed dẫn đầu)",
      random: "random (Ngẫu nhiên có kiểm soát)",
      default: "default (Chỉ cấu hình c00 mặc định)",
    };
    const RULE_LABELS = {
      epsilon_constraint: "epsilon_constraint (Tối ưu tốc độ/VRAM trong ngưỡng giữ chất lượng ε)",
      weighted: "weighted (Tổng trọng số cân bằng U)",
      tchebycheff: "tchebycheff (Min-max khoảng cách Chebyshev)",
      topsis: "topsis (Gần điểm lý tưởng TOPSIS)",
      knee: "knee (Điểm khuỷu Pareto tự động)",
    };
    if (methodSel && resData.methods) {
      methodSel.innerHTML = resData.methods.map((m) => `<option value="${esc(m)}" ${m === "conformal_mf" ? "selected" : ""}>${esc(METHOD_LABELS[m] || m)}</option>`).join("");
    }
    if (ruleSel && resData.rules) {
      ruleSel.innerHTML = resData.rules.map((r) => `<option value="${esc(r)}" ${r === "epsilon_constraint" ? "selected" : ""}>${esc(RULE_LABELS[r] || r)}</option>`).join("");
    }
    if (cardSel) {
      const cards = resData.conformal_cards || [];
      cardSel.innerHTML = `<option value="">(Tự động hiệu chỉnh online khi không chọn thẻ)</option>` + cards.map((c) => `<option value="${esc(c.path)}">${esc(c.name)} (${esc(c.path)})</option>`).join("");
    }

    function selectedModelsList() {
      if (!modelSel || !modelSel.value) return [];
      if (modelSel.value === "__ALL_LOCAL_PT__") return allWeights.map((w) => w.path);
      return [modelSel.value];
    }

    function readPayload(confirmed) {
      const mode = document.getElementById("launch-mode").value;
      const modelsVal = selectedModelsList();
      const selectedMethod = methodSel.value || "conformal_mf";
      const bmMethods = Array.from(new Set([selectedMethod, "conformal_mf", "mf_no_conformal", "adaptive_full", "tpe_bo", "random"]));
      const cardVal = cardSel ? cardSel.value.trim() : "";
      const tolRaw = tolInput ? tolInput.value.trim() : "";
      const maxMemRaw = maxMemInput ? maxMemInput.value.trim() : "";
      return {
        mode,
        data: dataInput.value.trim(),
        device: document.getElementById("launch-device").value.trim() || "auto",
        space: spaceSel ? spaceSel.value : "core",
        method: selectedMethod,
        methods: mode === "benchmark" ? bmMethods : undefined,
        selection_rule: ruleSel.value,
        quality_tolerance: tolRaw !== "" ? Number(tolRaw) : undefined,
        max_memory: maxMemRaw !== "" ? Number(maxMemRaw) : undefined,
        conformal_vram_card: cardVal || undefined,
        detection_metric: document.getElementById("launch-metric").value,
        full_epochs: Number((epochsInput && epochsInput.value) || 5),
        budget_seconds: Number((budgetInput && budgetInput.value) || 600),
        seeds: seedsInput ? seedsInput.value.trim() : "17 29",
        weights: document.getElementById("launch-weights").value.trim(),
        output: document.getElementById("launch-output").value.trim(),
        models: modelsVal,
        confirmed: Boolean(confirmed),
      };
    }

    function updateBudgetAdvisory(p) {
      if (!advisoryEl) return;
      const nImgs = currentApproxImages();
      const seedsCount = Math.max(1, (p.seeds || "17").replace(/,/g, " ").split(/\s+/).filter(Boolean).length);
      const isCpu = String(p.device || "").toLowerCase() === "cpu";
      if (!nImgs || p.mode === "plan-search") {
        advisoryEl.innerHTML = "";
        return;
      }
      // Empirical baseline calibrated on RTX 3050 + Ultralytics YOLO (~30s/epoch per 1000 train images; ~5x on CPU)
      const secPerEpoch = Math.max(1.5, (nImgs / 1000) * (isCpu ? 150.0 : 30.0));
      const estOneSeed = 25 + secPerEpoch * Math.max(1, p.full_epochs);
      const warnThreshold = estOneSeed * Math.max(1.15, seedsCount * 0.5);
      const minRecommended = Math.round(estOneSeed * seedsCount * 1.5);
      if (p.budget_seconds < warnThreshold) {
        advisoryEl.innerHTML = `
          <div class="alert alert-warn" style="display:flex;justify-content:space-between;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <div>
              <strong>Cảnh báo nguy cơ hết giờ (Timeout):</strong> Bộ dữ liệu có ~<strong>${nImgs} ảnh train</strong>, chạy <strong>${p.full_epochs} epochs × ${seedsCount} seed</strong> ước tính cần tối thiểu khoảng <strong>${minRecommended} giây</strong> để kịp xác nhận trọn vẹn ít nhất vài cấu hình. Ngân sách hiện tại (${p.budget_seconds}s) rất dễ bị ngắt giữa chừng.
            </div>
            <button type="button" id="apply-safe-budget-btn" class="btn btn-sm">Áp dụng gợi ý (${minRecommended}s)</button>
          </div>
        `;
        const applyBtn = document.getElementById("apply-safe-budget-btn");
        if (applyBtn) {
          applyBtn.addEventListener("click", () => {
            if (budgetInput) budgetInput.value = String(minRecommended);
            updatePreview();
          });
        }
      } else {
        advisoryEl.innerHTML = "";
      }
    }

    function updatePreview() {
      const p = readPayload(false);
      const seedsList = (p.seeds || "17 29 43").replace(/,/g, " ").split(/\s+/).filter(Boolean);
      const weightsList = (p.weights || "0.6 0.25 0.15").replace(/,/g, " ").split(/\s+/).filter(Boolean);
      const parts = [
        "python -m research_pipeline.v2",
        p.mode,
        "--data", `"${p.data}"`,
        "--device", p.device,
        "--space", p.space,
        "--full-epochs", p.full_epochs,
        "--budget-seconds", p.budget_seconds,
        "--seeds", ...(seedsList.length ? seedsList : ["17", "29", "43"]),
        "--selection-rule", p.selection_rule,
        "--weights", ...(weightsList.length ? weightsList : ["0.6", "0.25", "0.15"]),
        "--detection-metric", p.detection_metric,
        "--output", `"${p.output}"`,
      ];
      if (p.mode === "benchmark") {
        const refBudget = Math.max(Number(p.budget_seconds || 300) * 2, 600);
        parts.push(
          "--methods", ...(p.methods || ["adaptive_full", "adaptive_ehvi", "random"]),
          "--search-seeds", "17", "29", "43",
          "--reference-budget-seconds", refBudget,
        );
      } else {
        parts.push("--method", p.method);
      }
      if (p.models.length) parts.push("--models", ...p.models);
      if (p.conformal_vram_card) parts.push("--conformal-vram-card", `"${p.conformal_vram_card}"`);
      if (typeof p.quality_tolerance === "number" && Number.isFinite(p.quality_tolerance)) parts.push("--quality-tolerance", String(p.quality_tolerance));
      if (typeof p.max_memory === "number" && Number.isFinite(p.max_memory)) parts.push("--max-memory", String(p.max_memory));
      previewEl.textContent = parts.join(" ");
      updateBudgetAdvisory(p);
    }

    document.querySelectorAll(".scenario-card").forEach((card) => {
      card.addEventListener("click", () => {
        const sc = card.getAttribute("data-scenario");
        document.querySelectorAll(".scenario-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        if (sc === "fast") {
          if (epochsInput) epochsInput.value = "3";
          if (budgetInput) budgetInput.value = "300";
          if (seedsInput) seedsInput.value = "17";
          if (spaceSel) spaceSel.value = "compact";
        } else if (sc === "balanced") {
          if (epochsInput) epochsInput.value = "5";
          if (budgetInput) budgetInput.value = "600";
          if (seedsInput) seedsInput.value = "17 29";
          if (spaceSel) spaceSel.value = "core";
        } else if (sc === "paper") {
          if (epochsInput) epochsInput.value = "10";
          if (budgetInput) budgetInput.value = "900";
          if (seedsInput) seedsInput.value = "17 29 43";
          if (spaceSel) spaceSel.value = "core";
        }
        updatePreview();
      });
    });

    document.querySelectorAll(".budget-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const v = btn.getAttribute("data-budget");
        if (v && budgetInput) {
          budgetInput.value = v;
          document.querySelectorAll(".budget-preset-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          updatePreview();
        }
      });
    });

    [
      "launch-dataset-select", "launch-data", "launch-device", "launch-mode",
      "launch-space", "launch-models", "launch-method", "launch-rule",
      "launch-metric", "launch-epochs", "launch-budget", "launch-seeds",
      "launch-weights", "launch-conformal-card", "launch-quality-tolerance",
      "launch-max-memory", "launch-output",
    ].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("input", updatePreview);
        el.addEventListener("change", updatePreview);
      }
    });
    updatePreview();

    if (inspectBtn) {
      inspectBtn.addEventListener("click", async () => {
        inspectOut.innerHTML = `<div class="subtitle">Đang quét kiểm tra dữ liệu và rò rỉ ảnh SHA-256 (chỉ đọc)...</div>`;
        try {
          const out = await apiPost("/api/inspect", { data: dataInput.value.trim(), device: document.getElementById("launch-device").value.trim() || "cpu" });
          const ds = out.dataset || {};
          if (ds.counts && typeof ds.counts.train === "number") {
            inspectedTrainCount = ds.counts.train;
            updatePreview();
          }
          const au = ds.audit || {};
          const leak = (au.cross_split_leakage && au.cross_split_leakage.has_leakage) ? "CẢNH BÁO: PHÁT HIỆN RÒ RỈ XUYÊN TẬP" : "An toàn (0 ảnh trùng xuyên tập)";
          const dup = (au.intra_split_duplicates && au.intra_split_duplicates.total_duplicate_files) || 0;
          const clsMap = (au.class_counts && au.class_counts.total) || au.class_counts;
          const cls = clsMap ? Object.entries(clsMap).map(([k, v]) => `${k}: ${v}`).join(", ") : "N/A";
          inspectOut.innerHTML = `
            <div class="alert ${ds.validation_ready ? "alert-success" : "alert-warn"}">
              <strong>Tác vụ:</strong> ${esc(ds.task)} | <strong>Số ảnh (train/val/test):</strong> ${esc((ds.counts && ds.counts.train) ?? "?")} / ${esc((ds.counts && ds.counts.val) ?? "?")} / ${esc((ds.counts && ds.counts.test) ?? "?")} |
              <strong>Ảnh trùng nội bộ:</strong> ${dup} | <strong>${esc(leak)}</strong><br/>
              <strong>Phân bố lớp:</strong> ${esc(cls)} | <strong>Tỷ lệ lệch lớp:</strong> ${au.imbalance_ratio ? fmt(au.imbalance_ratio, 2) + "x" : "N/A"}
            </div>
          `;
        } catch (err) {
          inspectOut.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
        }
      });
    }

    if (launchBtn) {
      launchBtn.addEventListener("click", async () => {
        const confirmBox = document.getElementById("launch-confirm-check");
        if (!confirmBox || !confirmBox.checked) {
          launchMsg.innerHTML = `<div class="alert alert-warn">Vui lòng đánh dấu ô xác nhận trước khi khởi chạy tiến trình.</div>`;
          return;
        }
        try {
          const res = await apiPost("/api/launch", readPayload(true));
          launchMsg.innerHTML = `<div class="alert alert-success">Đã khởi chạy tiến trình <strong class="mono">${esc(res.job_id)}</strong> (PID ${esc(res.pid)}). <a href="/monitor">Mở màn hình Theo dõi trực tiếp</a></div>`;
        } catch (err) {
          launchMsg.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
        }
      });
    }
  }

  // ---- View 4: Phân tích & Biên Pareto ----
  async function initAnalysis() {
    const sel = document.getElementById("analysis-run-select");
    const runsData = await apiGet("/api/runs");
    const runs = (runsData.runs || []).filter((r) => r.kind === "v2");
    sel.innerHTML = runs.length
      ? runs.map((r) => `<option value="${esc(r.rel_path)}">${esc(r.rel_path)} — ${esc(r.method)} (${esc(statusLabel(r.verification_status || r.phase))})</option>`).join("")
      : `<option value="">(Chưa tìm thấy lượt chạy v2 nào)</option>`;
    const pre = queryParam("run") || (runs[0] && runs[0].rel_path) || "";
    if (pre) sel.value = pre;

    let currentDetail = null;

    function updateSensitivity() {
      const outEl = document.getElementById("sensitivity-table-body");
      const wA = Number(document.getElementById("slider-wa").value);
      const wT = Number(document.getElementById("slider-wt").value);
      const wM = Number(document.getElementById("slider-wm").value);
      document.getElementById("val-wa").textContent = wA.toFixed(2);
      document.getElementById("val-wt").textContent = wT.toFixed(2);
      document.getElementById("val-wm").textContent = wM.toFixed(2);
      if (!currentDetail || !currentDetail.report || !currentDetail.protocol) return;

      const rep = currentDetail.report;
      const obj = currentDetail.protocol.objective || {};
      const tScale = Number(obj.time_scale || 600);
      const mScale = Number(obj.memory_scale || 1024);
      const sumW = Math.max(1e-9, wA + wT + wM);
      const nA = wA / sumW, nT = wT / sumW, nM = wM / sumW;
      const feasible = (rep.candidates || []).filter((c) => c.state === "confirmed_feasible");
      const lockedId = rep.decision && rep.decision.selected_config_id;

      const scored = feasible.map((c) => {
        const q = Number(c.quality_mean || 0);
        const tRatio = Number(c.time_mean || 0) / tScale;
        const mRatio = Number(c.memory_max || 0) / mScale;
        const uPost = nA * q - nT * tRatio - nM * mRatio;
        return { ...c, post_utility: uPost };
      }).sort((a, b) => b.post_utility - a.post_utility);

      outEl.innerHTML = scored.length
        ? scored.map((c, idx) => `
          <tr>
            <td class="num">${idx + 1}</td>
            <td class="mono">c${String(c.index).padStart(2, "0")} ${c.config_id === lockedId ? "★ (Đã khóa)" : ""}</td>
            <td class="num">${fmt(c.quality_mean, 4)}</td>
            <td class="num">${fmt(c.time_mean, 2)}s</td>
            <td class="num">${fmt(c.memory_max, 1)} MiB</td>
            <td class="num">${fmt(c.utility, 6)}</td>
            <td class="num"><strong>${fmt(c.post_utility, 6)}</strong></td>
          </tr>
        `).join("")
        : `<tr><td colspan="7" class="subtitle">Không có cấu hình khả thi xác nhận đầy đủ để mô phỏng độ nhạy.</td></tr>`;
    }

    document.querySelectorAll(".weight-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const raw = (btn.getAttribute("data-preset-w") || "").split(",").map(Number);
        if (raw.length === 3 && raw.every(Number.isFinite)) {
          document.getElementById("slider-wa").value = String(raw[0]);
          document.getElementById("slider-wt").value = String(raw[1]);
          document.getElementById("slider-wm").value = String(raw[2]);
          document.querySelectorAll(".weight-preset-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          updateSensitivity();
        }
      });
    });

    async function loadSelected() {
      const target = sel.value;
      if (!target) return;
      const [detail, evalPayload] = await Promise.all([
        apiGet(`/api/run?path=${encodeURIComponent(target)}`),
        apiGet(`/api/evaluation?path=${encodeURIComponent(target)}`),
      ]);
      currentDetail = detail;
      const rep = detail.report || {};
      const prot = detail.protocol || {};
      const rec = rep.recommendation || {};
      const dec = rep.decision || {};
      const detMetric = rep.detection_metric || prot.detection_metric || null;
      const objWeights = prot.objective && prot.objective.weights;
      if (Array.isArray(objWeights) && objWeights.length === 3) {
        const rawW = objWeights.map((w) => Math.max(0, Number.isFinite(Number(w)) ? Number(w) : 0));
        const wSum = rawW.reduce((a, b) => a + b, 0);
        if (wSum > 0) {
          const sWa = document.getElementById("slider-wa");
          const sWt = document.getElementById("slider-wt");
          const sWm = document.getElementById("slider-wm");
          if (sWa) sWa.value = String(rawW[0] / wSum);
          if (sWt) sWt.value = String(rawW[1] / wSum);
          if (sWm) sWm.value = String(rawW[2] / wSum);
        }
      }

      document.getElementById("analysis-kpis").innerHTML = `
        <div class="card"><div class="kpi-label">Cấu hình được chọn</div><div class="kpi-value" style="font-size:1.1rem;">${esc(dec.selected_config_id || "Không có")}</div><div class="kpi-sub"><span class="badge ${badgeClass(dec.verification_status)}">${esc(statusLabel(dec.verification_status))}</span></div></div>
        <div class="card"><div class="kpi-label">Trung bình Validation</div><div class="kpi-value">${fmt(rec.validation_mean, 4)}</div><div class="kpi-sub">Độ lệch SD: ${fmt(rec.validation_sd_not_ci, 4)} · Độ hữu ích U: ${fmt(rec.utility, 6)}</div></div>
        <div class="card"><div class="kpi-label">Hypervolume 3D</div><div class="kpi-value">${fmt(detail.hypervolume_3d, 4)}</div><div class="kpi-sub">AUARC: ${fmt(detail.anytime && detail.anytime.auarc, 4)}</div></div>
        <div class="card"><div class="kpi-label">Sai số Surrogate (LOOCV MAE)</div><div class="kpi-value">${detail.surrogate_loocv && detail.surrogate_loocv.available ? fmt(detail.surrogate_loocv.loocv_mae, 4) : "—"}</div><div class="kpi-sub">Kernel: ${esc((detail.surrogate_loocv && detail.surrogate_loocv.kernel) || "Không dùng")}</div></div>
      `;

      const specEl = document.getElementById("analysis-selected-spec");
      if (specEl) {
        const selCand = (rep.candidates || []).find((c) => c.index === dec.selected_index);
        if (selCand && selCand.config) {
          const cfg = selCand.config;
          const effBatch = Number(cfg.batch || 1) * Number(cfg.accumulation || 1);
          specEl.innerHTML = `
            <h2>Cấu hình tối ưu được chọn — c${String(selCand.index).padStart(2, "0")} <span class="mono">(${esc(selCand.config_id)})</span></h2>
            <div class="spec-grid">
              <div class="spec-item"><div class="spec-label">Mô hình</div><div class="spec-val">${esc(baseName(cfg.model))}</div></div>
              <div class="spec-item"><div class="spec-label">Kích thước ảnh</div><div class="spec-val">${esc(cfg.imgsz)} px</div></div>
              <div class="spec-item"><div class="spec-label">Batch × Tích lũy</div><div class="spec-val">${esc(cfg.batch)} × ${esc(cfg.accumulation)} (eff ${effBatch})</div></div>
              <div class="spec-item"><div class="spec-label">Độ chính xác / Opt</div><div class="spec-val">${esc(String(cfg.precision || "fp32").toUpperCase())} / ${esc(cfg.optimizer)}</div></div>
              <div class="spec-item"><div class="spec-label">Learning Rate / WD</div><div class="spec-val">${esc(cfg.lr0)} / ${esc(cfg.weight_decay)}</div></div>
              <div class="spec-item"><div class="spec-label">Freeze / Epochs</div><div class="spec-val">${esc(cfg.freeze ?? 0)} block / ${esc(prot.full_epochs)} ep</div></div>
              <div class="spec-item"><div class="spec-label">Thời gian TB / seed</div><div class="spec-val">${fmt(selCand.time_mean, 2)}s</div></div>
              <div class="spec-item"><div class="spec-label">VRAM / RAM đỉnh</div><div class="spec-val">${fmt(selCand.memory_max, 1)} MiB</div></div>
            </div>
          `;
        } else {
          specEl.innerHTML = `
            <h2>Chưa có cấu hình được chọn</h2>
            <div class="alert alert-warn" style="margin-bottom:0;">
              Lượt chạy này kết thúc với trạng thái <strong>${esc(statusLabel(dec.verification_status))}</strong> nên chưa khóa được cấu hình tối ưu nào. Hãy kiểm tra bảng cấu hình bên dưới hoặc xem nhật ký ở <a href="/monitor?run=${encodeURIComponent(target)}">Theo dõi trực tiếp</a>.
            </div>
          `;
        }
      }

      const candTableBody = document.getElementById("analysis-candidates-body");
      if (candTableBody) {
        const paretoSet = new Set(dec.pareto || []);
        const cands = rep.candidates || [];
        candTableBody.innerHTML = cands.length
          ? cands.map((c) => {
            const cfg = c.config || {};
            const isSel = c.index === dec.selected_index;
            const isPar = paretoSet.has(c.config_id);
            const tag = `${isSel ? "★ " : ""}${isPar ? "Pareto" : ""}`;
            return `
              <tr>
                <td class="mono">c${String(c.index).padStart(2, "0")} ${tag ? `<span class="badge ${isSel ? "badge-info" : "badge-success"}">${esc(tag.trim())}</span>` : ""}</td>
                <td>${esc(baseName(cfg.model))}</td>
                <td class="num">${esc(cfg.imgsz)}</td>
                <td class="num">${esc(cfg.batch)} × ${esc(cfg.accumulation)}</td>
                <td>${esc(String(cfg.precision || "").toUpperCase())} / ${esc(cfg.optimizer)}</td>
                <td class="num">${esc(cfg.lr0)}</td>
                <td><span class="badge ${badgeClass(c.state)}" title="${esc(c.state)}">${esc(statusLabel(c.state))}</span></td>
                <td class="num">${fmt(c.quality_mean, 4)}</td>
                <td class="num">${c.time_mean !== null && c.time_mean !== undefined ? fmt(c.time_mean, 2) + "s" : "—"}</td>
                <td class="num">${c.memory_max !== null && c.memory_max !== undefined ? fmt(c.memory_max, 1) + " MiB" : "—"}</td>
                <td class="num">${fmt(c.utility, 5)}</td>
              </tr>
            `;
          }).join("")
          : `<tr><td colspan="11" class="subtitle">Chưa có cấu hình nào được ghi nhận.</td></tr>`;
      }

      window.NCKHCharts.renderParetoChart(document.getElementById("chart-pareto"), rep.candidates || [], dec, detMetric);
      window.NCKHCharts.renderLearningCurvesChart(document.getElementById("chart-learning"), (detail.state && detail.state.results) || {}, dec.selected_index, detMetric);
      window.NCKHCharts.renderSurrogateTraceChart(document.getElementById("chart-surrogate"), rep.acquisition_history || (detail.state && detail.state.acquisition_history) || [], rep.candidates || []);
      window.NCKHCharts.renderAnytimeChart(document.getElementById("chart-anytime"), detail.anytime);
      updateSensitivity();

      // Đánh giá độc lập trên tập Test
      const ev = (evalPayload && evalPayload.evaluation) || detail.evaluation;
      const evalSummaryEl = document.getElementById("eval-summary-box");
      if (!ev) {
        evalSummaryEl.innerHTML = `<div class="subtitle">Lượt chạy này chưa được đánh giá trên tập Test độc lập (hãy chạy lệnh evaluate-v2).</div>`;
        window.NCKHCharts.renderPerClassChart(document.getElementById("chart-per-class"), []);
        window.NCKHCharts.renderCurveChart(document.getElementById("chart-pr"), [], [], [], "Recall", "Precision");
        window.NCKHCharts.renderCurveChart(document.getElementById("chart-f1"), [], [], [], "Độ tin cậy", "F1");
        window.NCKHCharts.renderConfusionMatrix(document.getElementById("chart-cm"), null);
      } else {
        const task = prot.task || ev.task || "detection";
        const lat = ev.latency || {};
        const fpsVal = metricScalar(lat.fps) ?? metricScalar(lat.mean_fps);
        const msVal = metricScalar(lat.total_ms_per_image) ?? metricScalar(lat.mean_total_ms_per_image);
        const vvt = ev.validation_vs_test || {};
        const mss = ev.multi_seed_summary || {};
        const testQuality = metricScalar(mss.quality) ?? metricScalar(ev.quality);
        const testMap50 = metricScalar(mss.map50) ?? metricScalar(ev.map50);
        const testMap5095 = metricScalar(mss.map50_95) ?? metricScalar(ev.map50_95);
        const testPrec = metricScalar(mss.precision) ?? metricScalar(ev.precision);
        const testRec = metricScalar(mss.recall) ?? metricScalar(ev.recall);
        const testF1 = metricScalar(mss.f1) ?? metricScalar(ev.f1);
        const testAcc = metricScalar(mss.accuracy) ?? metricScalar(ev.accuracy) ?? testQuality;
        const testBleu4 = metricScalar(mss.bleu4) ?? metricScalar(ev.bleu4);
        const testChrf = metricScalar(mss.chrf) ?? metricScalar(ev.chrf);
        const qDef = prot.quality_definition || vvt.quality_definition || task;
        let kpiCardsHtml = "";
        if (task === "classification") {
          kpiCardsHtml = `
            <div class="card"><div class="kpi-label">Độ chính xác Test</div><div class="kpi-value">${fmt(testAcc, 4)}</div><div class="kpi-sub">Val: ${fmt(vvt.validation_mean, 4)} · Chênh lệch: ${fmt(vvt.generalization_gap_test_minus_val, 4)}</div></div>
            <div class="card"><div class="kpi-label">Chất lượng Test</div><div class="kpi-value">${fmt(testQuality, 4)}</div><div class="kpi-sub">Chỉ số: ${esc(qDef)}</div></div>
          `;
        } else if (task === "captioning") {
          kpiCardsHtml = `
            <div class="card"><div class="kpi-label">Chất lượng Test</div><div class="kpi-value">${fmt(testQuality, 4)}</div><div class="kpi-sub">Val: ${fmt(vvt.validation_mean, 4)} · Chênh lệch: ${fmt(vvt.generalization_gap_test_minus_val, 4)}</div></div>
            <div class="card"><div class="kpi-label">Test BLEU-4 &amp; chrF</div><div class="kpi-value">${fmt(testBleu4, 2)} / ${fmt(testChrf, 2)}</div><div class="kpi-sub">Chỉ số: ${esc(qDef)}</div></div>
          `;
        } else {
          kpiCardsHtml = `
            <div class="card"><div class="kpi-label">Test mAP50</div><div class="kpi-value">${fmt(testMap50, 4)}</div><div class="kpi-sub">Val: ${fmt(vvt.validation_mean, 4)} · Chênh lệch: ${fmt(vvt.generalization_gap_test_minus_val, 4)}</div></div>
            <div class="card"><div class="kpi-label">Test mAP50-95</div><div class="kpi-value">${fmt(testMap5095, 4)}</div><div class="kpi-sub">P=${fmt(testPrec, 3)} R=${fmt(testRec, 3)} F1=${fmt(testF1, 3)}</div></div>
          `;
        }
        evalSummaryEl.innerHTML = `
          <div class="grid grid-4">
            ${kpiCardsHtml}
            <div class="card"><div class="kpi-label">Độ trễ suy luận</div><div class="kpi-value">${msVal !== null ? fmt(msVal, 2) + " ms" : "—"}</div><div class="kpi-sub">rect=False · ${esc(lat.precision || "fp32")} · batch=${esc(lat.batch_size ?? 1)}</div></div>
            <div class="card"><div class="kpi-label">Tốc độ xử lý</div><div class="kpi-value">${fpsVal !== null ? fmt(fpsVal, 1) + " FPS" : "—"}</div><div class="kpi-sub">Warmup: ${esc(lat.warmup_batches ?? "—")} batch</div></div>
          </div>
        `;
        const perClassRows = (Array.isArray(ev.per_class_summary) && ev.per_class_summary.length)
          ? ev.per_class_summary
          : (ev.per_class || []);
        window.NCKHCharts.renderPerClassChart(document.getElementById("chart-per-class"), perClassRows);
        const pr = ev.pr_curve || {};
        window.NCKHCharts.renderCurveChart(document.getElementById("chart-pr"), pr.recall_grid, pr.mean_precision, pr.per_class, "Recall", "Precision");
        const f1c = ev.f1_curve || {};
        window.NCKHCharts.renderCurveChart(document.getElementById("chart-f1"), f1c.confidence_grid, f1c.mean_f1, f1c.per_class, "Độ tin cậy", "F1");
        window.NCKHCharts.renderConfusionMatrix(document.getElementById("chart-cm"), ev.confusion_matrix);
      }
    }

    ["slider-wa", "slider-wt", "slider-wm"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", updateSensitivity);
    });
    sel.addEventListener("change", loadSelected);
    await loadSelected();
  }

  // ---- View 5: So sánh & Đối chứng thuật toán ----
  async function initBenchmark() {
    const sel = document.getElementById("benchmark-select");
    const runsData = await apiGet("/api/runs");
    const items = [
      ...(runsData.benchmarks || []).map((b) => ({ ...b, label: `[Đối chứng] ${b.rel_path || b.name}` })),
      ...(runsData.algorithm_studies || []).map((s) => ({ ...s, label: `[Nghiên cứu đa bộ] ${s.rel_path || s.name}` })),
    ];
    sel.innerHTML = items.length
      ? items.map((x) => `<option value="${esc(x.rel_path)}">${esc(x.label)}</option>`).join("")
      : `<option value="">(Chưa tìm thấy báo cáo đối chứng nào)</option>`;
    const pre = queryParam("path") || (items[0] && items[0].rel_path) || "";
    if (pre) sel.value = pre;

    async function loadBm() {
      const target = sel.value;
      const sumBody = document.getElementById("bm-summaries-body");
      const pairedBody = document.getElementById("bm-paired-body");
      if (!target) return;
      try {
        const res = await apiGet(`/api/benchmark?path=${encodeURIComponent(target)}`);
        if (res.kind === "benchmark") {
          const bm = res.benchmark || {};
          const sums = bm.summaries || [];
          window.NCKHCharts.renderBenchmarkChart(document.getElementById("chart-benchmark"), sums);
          sumBody.innerHTML = sums.map((s) => `
            <tr>
              <td><strong>${esc(s.method)}</strong></td>
              <td class="num">${s.n}</td>
              <td class="num">${fmt(s.feasibility_rate, 2)}</td>
              <td class="num">${fmt(s.conditional_regret_mean, 5)}</td>
              <td class="num">${fmt(s.conditional_quality_regret_mean, 5)}</td>
              <td class="num">${fmt(s.pareto_hit_rate, 2)}</td>
              <td class="num">${fmt(s.auarc_mean, 5)}</td>
              <td class="num">${fmt(s.online_seconds_mean, 1)}s</td>
              <td class="num">${typeof s.vram_violation_attempts_mean === "number" ? fmt(s.vram_violation_attempts_mean, 2) : "—"}</td>
              <td class="num">${typeof s.vram_violation_wasted_seconds_mean === "number" ? fmt(s.vram_violation_wasted_seconds_mean, 1) + "s" : "—"}</td>
              <td class="num">${typeof s.conformal_pruned_mean === "number" ? fmt(s.conformal_pruned_mean, 1) : "—"}</td>
            </tr>
          `).join("");
          pairedBody.innerHTML = (bm.outcomes || []).map((o) => `
            <tr>
              <td>${esc(o.method)}</td>
              <td class="num">${esc(o.search_seed)}</td>
              <td class="mono">${esc(o.selected || "Không có")}</td>
              <td class="num">${fmt(o.objective_regret, 5)}</td>
              <td class="num">${fmt(o.quality_regret, 5)}</td>
              <td class="num">${fmt(o.auarc, 5)}</td>
              <td class="num">${fmt(o.online_seconds, 1)}s</td>
            </tr>
          `).join("");
        } else {
          const st = res.algorithm_study || {};
          const studies = Array.isArray(st.studies) ? st.studies : [];
          const chartSums = [];
          const sumRowsHtml = [];
          const pairedRowsHtml = [];
          studies.forEach((study, sIdx) => {
            const rawName = String(study.study || study.study_id || `Nghiên cứu ${sIdx + 1}`);
            const shortName = rawName.split(/[\\/]/).filter(Boolean).slice(-2).join("/") || rawName;
            const domainTask = study.domain && study.domain.task ? ` (${study.domain.task})` : "";
            const studyLabel = `Nghiên cứu ${sIdx + 1}: ${shortName}${domainTask}`;
            if (studies.length > 1) {
              sumRowsHtml.push(`<tr class="study-group-row"><td colspan="11"><strong>${esc(studyLabel)}</strong></td></tr>`);
            }
            const outcomesByMethod = {};
            (study.outcomes || []).forEach((o) => {
              (outcomesByMethod[o.method] = outcomesByMethod[o.method] || []).push(o);
            });
            (study.method_summaries || []).forEach((s) => {
              const mOuts = outcomesByMethod[s.method] || [];
              const qRegs = mOuts.map((o) => o.quality_regret).filter((v) => typeof v === "number" && Number.isFinite(v));
              const parHits = mOuts.map((o) => o.pareto_hit).filter((v) => typeof v === "boolean");
              const auarcs = mOuts.map((o) => o.auarc).filter((v) => typeof v === "number" && Number.isFinite(v));
              const viols = mOuts.map((o) => o.vram_violation_attempt_count).filter((v) => typeof v === "number" && Number.isFinite(v));
              const wastes = mOuts.map((o) => o.vram_violation_wasted_seconds).filter((v) => typeof v === "number" && Number.isFinite(v));
              const pruns = mOuts.map((o) => o.conformal_pruned_count).filter((v) => typeof v === "number" && Number.isFinite(v));
              const meanArr = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
              const row = {
                method: studies.length > 1 ? `S${sIdx + 1}: ${s.method}` : s.method,
                n: s.search_repetitions,
                feasibility_rate: s.feasibility_rate,
                conditional_regret_mean: s.conditional_objective_regret && s.conditional_objective_regret.mean,
                conditional_quality_regret_mean: meanArr(qRegs),
                pareto_hit_rate: parHits.length ? parHits.filter(Boolean).length / parHits.length : null,
                auarc_mean: meanArr(auarcs),
                online_seconds_mean: s.online_seconds && s.online_seconds.mean,
                vram_violation_attempts_mean: meanArr(viols),
                vram_violation_wasted_seconds_mean: meanArr(wastes),
                conformal_pruned_mean: meanArr(pruns),
              };
              chartSums.push(row);
              sumRowsHtml.push(`
                <tr>
                  <td><strong>${esc(s.method)}</strong></td>
                  <td class="num">${esc(row.n)}</td>
                  <td class="num">${fmt(row.feasibility_rate, 2)}</td>
                  <td class="num">${fmt(row.conditional_regret_mean, 5)}</td>
                  <td class="num">${fmt(row.conditional_quality_regret_mean, 5)}</td>
                  <td class="num">${fmt(row.pareto_hit_rate, 2)}</td>
                  <td class="num">${fmt(row.auarc_mean, 5)}</td>
                  <td class="num">${fmt(row.online_seconds_mean, 1)}s</td>
                  <td class="num">${row.vram_violation_attempts_mean !== null ? fmt(row.vram_violation_attempts_mean, 2) : "—"}</td>
                  <td class="num">${row.vram_violation_wasted_seconds_mean !== null ? fmt(row.vram_violation_wasted_seconds_mean, 1) + "s" : "—"}</td>
                  <td class="num">${row.conformal_pruned_mean !== null ? fmt(row.conformal_pruned_mean, 1) : "—"}</td>
                </tr>
              `);
            });
            const pairs = study.paired_comparisons || [];
            if (studies.length > 1) {
              pairedRowsHtml.push(`<tr class="study-group-row"><td colspan="7"><strong>${esc(studyLabel)}</strong></td></tr>`);
            }
            pairs.forEach((p) => {
              const m = (p.metrics && p.metrics.objective_regret) || {};
              const w = m.wilcoxon || {};
              const sf = m.sign_flip || {};
              const pairedN = m.paired_seed_count ?? (m.candidate_minus_baseline && m.candidate_minus_baseline.n) ?? "—";
              const holmP = m.holm && (m.holm.adjusted_p_value ?? m.holm.p_adjusted);
              pairedRowsHtml.push(`<tr><td>${esc(p.candidate)} vs ${esc(p.baseline)}</td><td class="num">${esc(pairedN)}</td><td class="mono">Wilcoxon p=${fmt(w.p_value, 4)}</td><td class="num">SignFlip p=${fmt(sf.p_value, 4)}</td><td class="num">Holm p=${fmt(holmP, 4)}</td><td class="num">—</td><td>${esc(m.validity_statement || p.paired_nonparametric_scope || "")}</td></tr>`);
            });
          });
          window.NCKHCharts.renderBenchmarkChart(document.getElementById("chart-benchmark"), chartSums);
          sumBody.innerHTML = sumRowsHtml.join("") || `<tr><td colspan="11">Không có dữ liệu tổng hợp theo phương pháp.</td></tr>`;
          pairedBody.innerHTML = pairedRowsHtml.join("") || `<tr><td colspan="7">Không có kiểm định ghép cặp.</td></tr>`;
        }
      } catch (err) {
        sumBody.innerHTML = `<tr><td colspan="11">${esc(err.message)}</td></tr>`;
      }
    }

    sel.addEventListener("change", loadBm);
    await loadBm();
  }

  // ---- View 6: Kiểm toán & Xuất báo cáo NCKH ----
  async function initExport() {
    const sel = document.getElementById("export-run-select");
    const outInput = document.getElementById("export-output-dir");
    const langSel = document.getElementById("export-lang");
    const auditBtn = document.getElementById("btn-run-audit");
    const auditOut = document.getElementById("audit-result");
    const exportBtn = document.getElementById("btn-run-export");
    const exportOut = document.getElementById("export-result");
    const previewEl = document.getElementById("artifact-preview");

    const runsData = await apiGet("/api/runs");
    const allTargets = [
      ...(runsData.runs || []).filter((r) => r.kind === "v2"),
      ...(runsData.benchmarks || []),
      ...(runsData.algorithm_studies || []),
    ];
    sel.innerHTML = allTargets.length
      ? allTargets.map((r) => `<option value="${esc(r.rel_path)}">${esc(r.rel_path || r.name)} (${esc(kindLabel(r.kind))})</option>`).join("")
      : `<option value="">(Chưa có lượt chạy khả dụng)</option>`;
    const pre = queryParam("run") || (allTargets[0] && allTargets[0].rel_path) || "";
    if (pre) sel.value = pre;

    function suggestOut() {
      const name = (sel.value || "study").replace(/[^a-zA-Z0-9_-]+/g, "_");
      outInput.value = `paper_exports/${name}_${Date.now()}`;
    }
    suggestOut();
    sel.addEventListener("change", suggestOut);

    if (auditBtn) {
      auditBtn.addEventListener("click", async () => {
        auditOut.innerHTML = `<div class="subtitle">Đang kiểm toán nhật ký băm chuỗi SHA-256 và checkpoint...</div>`;
        try {
          const res = await apiPost("/api/audit", { run: sel.value });
          const checksHtml = (res.checks || []).map((c) => `<li><span class="badge ${c.passed ? "badge-success" : "badge-danger"}">${c.passed ? "ĐẠT" : "LỖI"}</span> <strong>${esc(c.name)}</strong>: ${esc(c.detail)}</li>`).join("");
          const verdictCls = res.ok ? "alert-success" : "alert-danger";
          const verdictText = res.ok ? "ĐẠT KIỂM TOÁN (PASS)" : "KHÔNG ĐẠT KIỂM TOÁN (FAIL)";
          auditOut.innerHTML = `<div class="alert ${verdictCls}"><strong>${verdictText}</strong> (${esc(res.kind)}, đã kiểm tra ${res.checked_attempts} mục)<ul>${checksHtml}</ul></div>`;
        } catch (err) {
          auditOut.innerHTML = `<div class="alert alert-danger">KHÔNG ĐẠT KIỂM TOÁN: ${esc(err.message)}</div>`;
        }
      });
    }

    if (exportBtn) {
      exportBtn.addEventListener("click", async () => {
        exportOut.innerHTML = `<div class="subtitle">Đang kiểm toán và xuất bảng LaTeX, CSV cùng biểu đồ vector SVG...</div>`;
        try {
          const res = await apiPost("/api/paper-export", {
            run: sel.value,
            output: outInput.value.trim(),
            lang: langSel.value,
          });
          const links = (res.files || []).map((f) => `
            <div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">
              <button type="button" class="btn btn-sm preview-file-btn" data-url="${esc(f.url)}" data-name="${esc(f.name)}">Xem trước</button>
              <a class="btn btn-sm" href="${esc(f.url)}&download=1">Tải xuống</a>
              <span class="mono">${esc(f.name)}</span>
            </div>
          `).join("");
          exportOut.innerHTML = `<div class="alert alert-success">Đã xuất thành công tại <span class="mono">${esc(res.output_rel)}</span></div>${links}`;
          exportOut.querySelectorAll(".preview-file-btn").forEach((btn) => {
            btn.addEventListener("click", async () => {
              const url = btn.getAttribute("data-url");
              const name = btn.getAttribute("data-name");
              if (name && name.toLowerCase().endsWith(".svg")) {
                previewEl.innerHTML = `<img src="${esc(url)}" alt="${esc(name)}" style="max-width:100%;height:auto;display:block;" />`;
                return;
              }
              const r = await fetch(url);
              const txt = await r.text();
              previewEl.innerHTML = `<pre class="log-box">${esc(txt)}</pre>`;
            });
          });
        } catch (err) {
          exportOut.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    const page = document.body.getAttribute("data-page");
    if (page === "overview") initOverview();
    else if (page === "monitor") initMonitor();
    else if (page === "launcher") initLauncher();
    else if (page === "analysis") initAnalysis();
    else if (page === "benchmark") initBenchmark();
    else if (page === "export") initExport();
  });
})();
