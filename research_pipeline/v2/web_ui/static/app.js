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

  function badgeClass(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("optimal") || s === "complete" || s === "confirmed_feasible" || s === "ok") return "badge-success";
    if (s.includes("best") || s === "search" || s === "prepared") return "badge-info";
    if (s.includes("no_verified") || s.includes("partial") || s.includes("interrupted")) return "badge-warn";
    if (s.includes("fail") || s.includes("oom") || s.includes("contended") || s.includes("infeasible")) return "badge-danger";
    return "badge-muted";
  }

  function statusLabel(status) {
    const raw = String(status || "");
    const s = raw.toLowerCase();
    if (s === "empirical_optimal") return "TỐI ƯU THỰC NGHIỆM";
    if (s === "observed_best") return "TỐT NHẤT QUAN SÁT";
    if (s === "confirmed_feasible") return "ĐẠT KHẢ THI ĐẦY ĐỦ";
    if (s === "complete") return "HOÀN TẤT";
    if (s === "search") return "ĐANG TÌM KIẾM";
    if (s === "prepared") return "ĐÃ LẬP KẾ HOẠCH";
    if (s === "no_verified_feasible") return "CHƯA CÓ CẤU HÌNH KHẢ THI";
    if (s === "observed_required_seed_oom") return "TRÀN VRAM (OOM)";
    if (s === "confirmed_constraint_failure") return "VI PHẠM RÀNG BUỘC";
    if (s === "infeasible") return "KHÔNG KHẢ THI";
    if (s === "interrupted") return "ĐÃ TẠM DỪNG";
    return raw || "—";
  }

  function kindLabel(kind) {
    if (kind === "v2") return "Tìm kiếm v2";
    if (kind === "v2_deployment") return "Triển khai v2";
    if (kind === "research") return "Nghiên cứu v1";
    if (kind === "legacy_training") return "Huấn luyện cũ";
    if (kind === "benchmark") return "Đối chứng Benchmark";
    if (kind === "algorithm_study") return "Nghiên cứu đa bộ dữ liệu";
    return kind || "—";
  }

  function queryParam(name) {
    return new URLSearchParams(window.location.search).get(name);
  }

  // ---- View 1: Tổng quan & Phần cứng ----
  async function initOverview() {
    const hwEl = document.getElementById("hw-cards");
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
        const ramPct = Math.min(100, Math.round(hw.ram_percent || 0));
        const gpuCard = gpu
          ? `<div class="card"><div class="kpi-label">Card đồ họa (GPU)</div><div class="kpi-value" style="font-size:1.05rem;">${esc(gpu.name)}</div><div class="kpi-sub">VRAM: ${fmt(gpu.vram_used_mb, 0)} / ${fmt(gpu.vram_total_mb, 0)} MiB · Nhiệt: ${gpu.temperature_c !== null ? fmt(gpu.temperature_c, 0) + "°C" : "N/A"} · Tải: ${gpu.gpu_utilization_percent !== null ? fmt(gpu.gpu_utilization_percent, 0) + "%" : "N/A"}</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${vramPct}%"></div></div></div>`
          : `<div class="card"><div class="kpi-label">Chế độ tính toán</div><div class="kpi-value">CPU</div><div class="kpi-sub">Không có CUDA; torch ${esc(hw.torch_version || "N/A")}</div></div>`;
        hwEl.innerHTML = `
          ${gpuCard}
          <div class="card"><div class="kpi-label">Vi xử lý (CPU)</div><div class="kpi-value">${fmt(hw.cpu_percent, 1)}%</div><div class="kpi-sub">${hw.logical_cpus} luồng xử lý · Chế độ ${esc(hw.mode.toUpperCase())}</div></div>
          <div class="card"><div class="kpi-label">Bộ nhớ hệ thống (RAM)</div><div class="kpi-value">${fmt(hw.ram_used_mb, 0)} MiB</div><div class="kpi-sub">Còn trống: ${fmt(hw.ram_available_mb, 0)} / ${fmt(hw.ram_total_mb, 0)} MiB (${fmt(hw.ram_percent, 1)}%)</div><div class="hw-bar progress-track"><div class="progress-fill" style="width:${ramPct}%"></div></div></div>
          <div class="card"><div class="kpi-label">Dung lượng ổ đĩa trống</div><div class="kpi-value">${fmt(hw.disk_free_mb / 1024, 1)} GB</div><div class="kpi-sub">Tổng dung lượng: ${fmt(hw.disk_total_mb / 1024, 1)} GB</div></div>
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
      runsBody.innerHTML = filtered.map((r) => `
        <tr>
          <td class="mono">${esc(r.rel_path || r.name)}</td>
          <td>${esc(kindLabel(r.kind))}</td>
          <td>${esc(r.task)}</td>
          <td>${esc(r.method)}</td>
          <td><span class="badge ${badgeClass(r.verification_status || r.phase)}" title="${esc(r.verification_status || r.phase)}">${esc(statusLabel(r.verification_status || r.phase))}</span></td>
          <td class="mono">${esc(r.selected_config_id || "—")}</td>
          <td class="num">${fmt(r.best_quality, 4)}</td>
          <td class="num">${fmt(r.utility, 4)}</td>
          <td>
            <a class="btn btn-sm" href="/analysis?run=${encodeURIComponent(r.rel_path)}">Phân tích</a>
            <a class="btn btn-sm" href="/monitor?run=${encodeURIComponent(r.rel_path)}">Theo dõi</a>
            <a class="btn btn-sm" href="/export?run=${encodeURIComponent(r.rel_path)}">Xuất</a>
          </td>
        </tr>
      `).join("");
    }

    async function loadRuns() {
      try {
        const data = await apiGet("/api/runs");
        cachedRuns = data.runs || [];
        renderRuns();
        const bms = [...(data.benchmarks || []), ...(data.algorithm_studies || [])];
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
    const stopBtn = document.getElementById("safe-stop-btn");
    const msgEl = document.getElementById("monitor-status-msg");
    const kpiEl = document.getElementById("monitor-kpis");
    const candBody = document.getElementById("monitor-candidates-body");
    const logEl = document.getElementById("monitor-worker-log");
    const jobsBody = document.getElementById("monitor-jobs-body");

    const runsData = await apiGet("/api/runs");
    const runs = runsData.runs || [];
    sel.innerHTML = `<option value="">-- Chọn lượt chạy cần theo dõi --</option>` + runs.map((r) => `<option value="${esc(r.rel_path)}">${esc(r.rel_path)} (${esc(statusLabel(r.phase))})</option>`).join("");
    const pre = queryParam("run") || (runs[0] && runs[0].rel_path) || "";
    if (pre) sel.value = pre;

    async function poll() {
      const target = sel.value;
      const url = target ? `/api/live?run=${encodeURIComponent(target)}` : "/api/live";
      try {
        const data = await apiGet(url);
        const jobs = data.jobs || [];
        jobsBody.innerHTML = jobs.length
          ? jobs.map((j) => `<tr><td class="mono">${esc(j.job_id)}</td><td>${esc(j.mode)}</td><td class="mono">${esc(j.pid)}</td><td><span class="badge ${j.alive ? "badge-info" : "badge-muted"}">${j.alive ? "ĐANG CHẠY" : "ĐÃ DỪNG (" + j.returncode + ")"}</span></td><td class="mono">${esc(j.output)}</td></tr>`).join("")
          : `<tr><td colspan="5" class="subtitle">Chưa có tiến trình nào được khởi chạy từ Web UI trong phiên này.</td></tr>`;

        const r = data.run;
        if (!r) {
          kpiEl.innerHTML = "";
          candBody.innerHTML = "";
          logEl.textContent = "Hãy chọn một lượt chạy ở phía trên để theo dõi tiến độ epoch và nhật ký worker.log.";
          return;
        }
        const lt = r.latest_trial || {};
        const epPct = lt.trial_epochs ? Math.min(100, Math.round(((lt.epochs_completed || 0) / lt.trial_epochs) * 100)) : 0;
        kpiEl.innerHTML = `
          <div class="card"><div class="kpi-label">Giai đoạn</div><div class="kpi-value" style="font-size:1.1rem;">${esc(statusLabel(r.phase))}</div><div class="kpi-sub">${esc(r.task)} · ${esc(r.method)}</div></div>
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
        logEl.textContent = (r.worker_log && r.worker_log.length) ? r.worker_log.join("\n") : "(Chưa có đầu ra trong worker.log)";
      } catch (err) {
        msgEl.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
      }
    }

    sel.addEventListener("change", poll);
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
    const budgetInput = document.getElementById("launch-budget");
    const inspectBtn = document.getElementById("btn-inspect-dataset");
    const inspectOut = document.getElementById("inspect-result");
    const previewEl = document.getElementById("cli-preview");
    const launchBtn = document.getElementById("btn-launch-job");
    const launchMsg = document.getElementById("launch-result");

    if (dsSel && resData.datasets) {
      dsSel.innerHTML = `<option value="">-- Chọn nhanh bộ dữ liệu có sẵn --</option>` + resData.datasets.map((d) => `<option value="${esc(d.path)}">${esc(d.name)} (${esc(d.kind)})</option>`).join("");
      dsSel.addEventListener("change", () => {
        if (dsSel.value) dataInput.value = dsSel.value;
        updatePreview();
      });
      if (resData.datasets[0] && !dataInput.value) dataInput.value = resData.datasets[0].path;
    }
    if (modelSel && resData.weights) {
      modelSel.innerHTML = `<option value="">(Dùng mặc định theo không gian cấu hình)</option>` + resData.weights.map((w) => `<option value="${esc(w.path)}">${esc(w.name)} — ${w.parameters ? (w.parameters / 1e6).toFixed(2) + "M tham số" : w.size_mb + " MB"}${w.gflops_640 ? " | " + w.gflops_640.toFixed(2) + " GFLOPs@640" : ""}</option>`).join("");
    }
    if (methodSel && resData.methods) {
      methodSel.innerHTML = resData.methods.map((m) => `<option value="${esc(m)}" ${m === "adaptive_full" ? "selected" : ""}>${esc(m)}</option>`).join("");
    }
    if (ruleSel && resData.rules) {
      ruleSel.innerHTML = resData.rules.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join("");
    }

    function readPayload(confirmed) {
      const mode = document.getElementById("launch-mode").value;
      const modelsVal = modelSel && modelSel.value ? [modelSel.value] : [];
      const selectedMethod = methodSel.value || "adaptive_full";
      const bmMethods = Array.from(new Set([selectedMethod, "adaptive_full", "adaptive_ehvi", "random"]));
      return {
        mode,
        data: dataInput.value.trim(),
        device: document.getElementById("launch-device").value.trim() || "auto",
        space: document.getElementById("launch-space").value,
        method: selectedMethod,
        methods: mode === "benchmark" ? bmMethods : undefined,
        selection_rule: ruleSel.value,
        detection_metric: document.getElementById("launch-metric").value,
        full_epochs: Number(document.getElementById("launch-epochs").value || 10),
        budget_seconds: Number((budgetInput && budgetInput.value) || 300),
        seeds: document.getElementById("launch-seeds").value.trim(),
        weights: document.getElementById("launch-weights").value.trim(),
        output: document.getElementById("launch-output").value.trim(),
        models: modelsVal,
        confirmed: Boolean(confirmed),
      };
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
      previewEl.textContent = parts.join(" ");
    }

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
      "launch-weights", "launch-output",
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
              sumRowsHtml.push(`<tr class="study-group-row"><td colspan="8"><strong>${esc(studyLabel)}</strong></td></tr>`);
            }
            (study.method_summaries || []).forEach((s) => {
              const row = {
                method: studies.length > 1 ? `S${sIdx + 1}: ${s.method}` : s.method,
                n: s.search_repetitions,
                feasibility_rate: s.feasibility_rate,
                conditional_regret_mean: s.conditional_objective_regret && s.conditional_objective_regret.mean,
                conditional_quality_regret_mean: null,
                pareto_hit_rate: null,
                auarc_mean: null,
                online_seconds_mean: s.online_seconds && s.online_seconds.mean,
              };
              chartSums.push(row);
              sumRowsHtml.push(`
                <tr>
                  <td><strong>${esc(s.method)}</strong></td>
                  <td class="num">${esc(row.n)}</td>
                  <td class="num">${fmt(row.feasibility_rate, 2)}</td>
                  <td class="num">${fmt(row.conditional_regret_mean, 5)}</td>
                  <td class="num">—</td>
                  <td class="num">—</td>
                  <td class="num">—</td>
                  <td class="num">${fmt(row.online_seconds_mean, 1)}s</td>
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
          sumBody.innerHTML = sumRowsHtml.join("") || `<tr><td colspan="8">Không có dữ liệu tổng hợp theo phương pháp.</td></tr>`;
          pairedBody.innerHTML = pairedRowsHtml.join("") || `<tr><td colspan="7">Không có kiểm định ghép cặp.</td></tr>`;
        }
      } catch (err) {
        sumBody.innerHTML = `<tr><td colspan="8">${esc(err.message)}</td></tr>`;
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
