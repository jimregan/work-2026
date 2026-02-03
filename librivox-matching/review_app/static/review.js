(function () {
  "use strict";

  let segmentsData = [];
  let annotations = {};  // key: "segIdx-opIdx" → classification string

  const alignBtn = document.getElementById("align-btn");
  const saveBtn = document.getElementById("save-btn");
  const exportBtn = document.getElementById("export-btn");
  const statusEl = document.getElementById("status");
  const segmentsEl = document.getElementById("segments");

  alignBtn.addEventListener("click", doAlign);
  saveBtn.addEventListener("click", doSave);
  exportBtn.addEventListener("click", doExport);

  // Close any open popup when clicking elsewhere
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".popup") && !e.target.closest(".diff-group.clickable")) {
      document.querySelectorAll(".popup").forEach(function (p) { p.remove(); });
    }
  });

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function doAlign() {
    const formData = new FormData();

    const vvFile = document.getElementById("vv-file").files[0];
    const etFile = document.getElementById("et-file").files[0];
    const vvText = document.getElementById("vv-text").value;
    const etText = document.getElementById("et-text").value;

    if (vvFile) {
      formData.append("vibevoice_file", vvFile);
    } else {
      formData.append("vibevoice_text", vvText);
    }

    if (etFile) {
      formData.append("etext_file", etFile);
    } else {
      formData.append("etext_text", etText);
    }

    alignBtn.disabled = true;
    setStatus("Aligning...");

    fetch("/align", { method: "POST", body: formData })
      .then(function (r) {
        if (!r.ok && r.headers.get("content-type") && r.headers.get("content-type").indexOf("json") === -1) {
          throw new Error("Server returned HTTP " + r.status);
        }
        return r.json();
      })
      .then(function (data) {
        alignBtn.disabled = false;
        if (data.error) {
          setStatus("Error: " + data.error);
          return;
        }
        segmentsData = data.segments;
        annotations = {};
        renderSegments();
        saveBtn.disabled = false;
        exportBtn.disabled = false;
        setStatus("Aligned " + segmentsData.length + " segments.");
      })
      .catch(function (err) {
        alignBtn.disabled = false;
        setStatus("Request failed: " + err);
      });
  }

  function renderSegments() {
    segmentsEl.innerHTML = "";
    segmentsData.forEach(function (seg) {
      const div = document.createElement("div");
      div.className = "segment" + (seg.boilerplate ? " boilerplate" : "");

      const header = document.createElement("div");
      header.className = "segment-header";
      header.innerHTML =
        "<span>#" + seg.index + " &nbsp; " +
        formatTime(seg.start) + " \u2013 " + formatTime(seg.end) + "</span>" +
        (seg.boilerplate ? '<span class="badge">boilerplate</span>' : "");
      div.appendChild(header);

      const body = document.createElement("div");
      body.className = "segment-body";

      if (seg.boilerplate) {
        body.innerHTML = '<div class="diff-line" style="font-style:italic;color:#999">' +
          escapeHtml(seg.vibevoice) + "</div>";
      } else {
        const diffLine = document.createElement("div");
        diffLine.className = "diff-line";
        seg.diff_ops.forEach(function (op, opIdx) {
          diffLine.appendChild(renderDiffOp(seg.index, opIdx, op));
        });
        body.appendChild(diffLine);
      }

      div.appendChild(body);
      segmentsEl.appendChild(div);
    });
  }

  function renderDiffOp(segIdx, opIdx, op) {
    const span = document.createElement("span");
    span.className = "diff-group";
    span.dataset.segIdx = segIdx;
    span.dataset.opIdx = opIdx;

    if (op.op === "equal") {
      op.et_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "equal-word";
        s.textContent = w + " ";
        span.appendChild(s);
      });
      return span;
    }

    // Non-equal: make clickable
    span.classList.add("clickable");
    span.addEventListener("click", function (e) {
      e.stopPropagation();
      togglePopup(span, segIdx, opIdx, op);
    });

    if (op.op === "replace") {
      op.et_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "et-word";
        s.textContent = w + " ";
        span.appendChild(s);
      });
      op.vv_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "vv-word-replace";
        s.textContent = w + " ";
        span.appendChild(s);
      });
    } else if (op.op === "insert") {
      op.vv_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "vv-word-insert";
        s.textContent = w + " ";
        span.appendChild(s);
      });
    } else if (op.op === "delete") {
      op.et_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "et-word-delete";
        s.textContent = w + " ";
        span.appendChild(s);
      });
    }

    // Show existing classification badge
    const key = segIdx + "-" + opIdx;
    if (annotations[key]) {
      appendBadge(span, annotations[key]);
    }

    return span;
  }

  function togglePopup(groupEl, segIdx, opIdx, op) {
    // Remove any existing popup
    document.querySelectorAll(".popup").forEach(function (p) { p.remove(); });

    const key = segIdx + "-" + opIdx;
    const current = annotations[key] || null;

    const popup = document.createElement("div");
    popup.className = "popup";

    var options = [
      { value: "asr", label: "ASR error" },
      { value: "reading", label: "Reading error" },
      { value: "normalisation", label: "Normalisation" },
    ];

    options.forEach(function (opt) {
      const btn = document.createElement("button");
      btn.className = "popup-item" + (current === opt.value ? " active" : "");
      btn.innerHTML = '<span class="dot ' + opt.value + '"></span>' + opt.label;
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        annotations[key] = opt.value;
        popup.remove();
        // Re-render badge on the group
        const existing = groupEl.querySelector(".class-badge");
        if (existing) existing.remove();
        appendBadge(groupEl, opt.value);
      });
      popup.appendChild(btn);
    });

    // Clear option
    if (current) {
      const clr = document.createElement("button");
      clr.className = "popup-item";
      clr.textContent = "\u2715 Clear";
      clr.addEventListener("click", function (e) {
        e.stopPropagation();
        delete annotations[key];
        popup.remove();
        const existing = groupEl.querySelector(".class-badge");
        if (existing) existing.remove();
      });
      popup.appendChild(clr);
    }

    groupEl.appendChild(popup);
  }

  function appendBadge(el, classification) {
    const badge = document.createElement("span");
    badge.className = "class-badge " + classification;
    var labels = { asr: "ASR", reading: "READ", normalisation: "NORM" };
    badge.textContent = labels[classification] || classification;
    el.appendChild(badge);
  }

  function buildAnnotationsList() {
    const list = [];
    Object.keys(annotations).forEach(function (key) {
      const parts = key.split("-");
      const segIdx = parseInt(parts[0], 10);
      const opIdx = parseInt(parts[1], 10);
      const seg = segmentsData[segIdx];
      var op = seg.diff_ops[opIdx];
      list.push({
        segment_index: segIdx,
        start: seg.start,
        end: seg.end,
        vibevoice: seg.vibevoice,
        etext: seg.etext,
        diff_op_index: opIdx,
        op: op.op,
        vv_words: op.vv_words,
        et_words: op.et_words,
        classification: annotations[key],
      });
    });
    list.sort(function (a, b) {
      return a.segment_index - b.segment_index || a.diff_op_index - b.diff_op_index;
    });
    return list;
  }

  function doSave() {
    const data = buildAnnotationsList();
    setStatus("Saving...");
    fetch("/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (res.error) {
          setStatus("Save error: " + res.error);
        } else {
          setStatus("Saved as " + res.filename);
        }
      })
      .catch(function (err) { setStatus("Save failed: " + err); });
  }

  function doExport() {
    const data = buildAnnotationsList();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
})();
