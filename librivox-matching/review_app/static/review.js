(function () {
  "use strict";

  let segmentsData = [];
  let annotations = {};  // key: "segIdx-opIdx" → { classification, normalised? }
  let pendingAnnotations = null;  // stored annotations to apply after alignment

  const alignBtn = document.getElementById("align-btn");
  const saveBtn = document.getElementById("save-btn");
  const exportBtn = document.getElementById("export-btn");
  const statusEl = document.getElementById("status");
  const segmentsEl = document.getElementById("segments");
  const loadInput = document.getElementById("load-file");

  alignBtn.addEventListener("click", doAlign);
  saveBtn.addEventListener("click", doSave);
  exportBtn.addEventListener("click", doExport);
  if (loadInput) {
    loadInput.addEventListener("change", doLoad);
  }

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
        // Apply pending annotations from a loaded file
        if (pendingAnnotations) {
          applyLoadedAnnotations(pendingAnnotations);
          pendingAnnotations = null;
        }
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
      div.dataset.segIdx = seg.index;

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
    span.className = "diff-group clickable";
    span.dataset.segIdx = segIdx;
    span.dataset.opIdx = opIdx;

    span.addEventListener("click", function (e) {
      e.stopPropagation();
      togglePopup(span, segIdx, opIdx, op);
    });

    if (op.op === "equal") {
      op.et_words.forEach(function (w) {
        const s = document.createElement("span");
        s.className = "equal-word";
        s.textContent = w + " ";
        span.appendChild(s);
      });
    } else if (op.op === "replace") {
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
      appendBadge(span, annotations[key].classification);
    }

    return span;
  }

  function togglePopup(groupEl, segIdx, opIdx, op) {
    // Remove any existing popup
    document.querySelectorAll(".popup").forEach(function (p) { p.remove(); });

    const key = segIdx + "-" + opIdx;
    const current = annotations[key] || null;
    const currentClass = current ? current.classification : null;

    const popup = document.createElement("div");
    popup.className = "popup";

    var options;
    if (op.op === "equal") {
      // Equal words get a reduced menu
      options = [
        { value: "reading", label: "Reading error" },
        { value: "normalisation", label: "Normalisation" },
      ];
    } else {
      options = [
        { value: "asr", label: "ASR error" },
        { value: "reading", label: "Reading error" },
        { value: "normalisation", label: "Normalisation" },
        { value: "punctuation", label: "Punctuation" },
        { value: "dialect", label: "Dialect" },
      ];
    }

    options.forEach(function (opt) {
      const btn = document.createElement("button");
      btn.className = "popup-item" + (currentClass === opt.value ? " active" : "");
      btn.innerHTML = '<span class="dot ' + opt.value + '"></span>' + opt.label;
      btn.addEventListener("click", function (e) {
        e.stopPropagation();

        // For reading/normalisation, show text input
        if (opt.value === "reading" || opt.value === "normalisation") {
          showNormInput(popup, groupEl, segIdx, opIdx, op, opt.value, current);
        } else {
          applyAnnotation(key, { classification: opt.value }, groupEl, segIdx, opIdx, op);
          popup.remove();
        }
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

  function showNormInput(popup, groupEl, segIdx, opIdx, op, classification, current) {
    // Remove existing input area if any
    var existingArea = popup.querySelector(".norm-input-area");
    if (existingArea) existingArea.remove();

    var area = document.createElement("div");
    area.className = "norm-input-area";

    var label = document.createElement("label");
    label.textContent = "Normalised text:";
    label.style.cssText = "display:block;font-size:0.75rem;margin:0.4rem 0.5rem 0.15rem;color:#666;";
    area.appendChild(label);

    var input = document.createElement("input");
    input.type = "text";
    input.className = "norm-input";

    // Pre-fill: reading → etext words, normalisation → VV words
    var prefill = "";
    if (current && current.normalised) {
      prefill = current.normalised;
    } else if (classification === "reading") {
      prefill = (op.et_words || []).join(" ");
    } else if (classification === "normalisation") {
      prefill = (op.vv_words || []).join(" ");
    }
    input.value = prefill;
    area.appendChild(input);

    var confirmBtn = document.createElement("button");
    confirmBtn.className = "popup-item";
    confirmBtn.textContent = "Apply";
    confirmBtn.style.cssText = "font-weight:700;color:#2563eb;";
    confirmBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      var key = segIdx + "-" + opIdx;
      var ann = { classification: classification };
      if (input.value.trim()) {
        ann.normalised = input.value.trim();
      }
      applyAnnotation(key, ann, groupEl, segIdx, opIdx, op);
      popup.remove();
    });
    area.appendChild(confirmBtn);

    popup.appendChild(area);
    input.focus();

    // Prevent popup from closing when clicking input
    input.addEventListener("click", function (e) { e.stopPropagation(); });
  }

  function applyAnnotation(key, ann, groupEl, segIdx, opIdx, op) {
    annotations[key] = ann;

    // Re-render badge on the group
    var existing = groupEl.querySelector(".class-badge");
    if (existing) existing.remove();
    appendBadge(groupEl, ann.classification);

    // Auto-propagate to matching (vv_words, et_words) pairs
    autopropagate(segIdx, opIdx, op, ann);
  }

  function autopropagate(sourceSegIdx, sourceOpIdx, sourceOp, ann) {
    var srcVV = (sourceOp.vv_words || []).join(" ");
    var srcET = (sourceOp.et_words || []).join(" ");

    var affectedSegments = {};

    segmentsData.forEach(function (seg) {
      if (seg.boilerplate) return;
      seg.diff_ops.forEach(function (op, opIdx) {
        if (seg.index === sourceSegIdx && opIdx === sourceOpIdx) return;
        var vv = (op.vv_words || []).join(" ");
        var et = (op.et_words || []).join(" ");
        if (vv === srcVV && et === srcET) {
          var k = seg.index + "-" + opIdx;
          annotations[k] = { classification: ann.classification };
          if (ann.normalised) {
            annotations[k].normalised = ann.normalised;
          }
          affectedSegments[seg.index] = true;
        }
      });
    });

    // Re-render affected segments
    Object.keys(affectedSegments).forEach(function (segIdxStr) {
      rerenderSegment(parseInt(segIdxStr, 10));
    });
  }

  function rerenderSegment(segIdx) {
    var seg = segmentsData[segIdx];
    if (!seg || seg.boilerplate) return;
    var segEl = segmentsEl.querySelector('.segment[data-seg-idx="' + segIdx + '"]');
    if (!segEl) return;
    var body = segEl.querySelector(".segment-body");
    if (!body) return;
    body.innerHTML = "";
    var diffLine = document.createElement("div");
    diffLine.className = "diff-line";
    seg.diff_ops.forEach(function (op, opIdx) {
      diffLine.appendChild(renderDiffOp(seg.index, opIdx, op));
    });
    body.appendChild(diffLine);
  }

  function appendBadge(el, classification) {
    const badge = document.createElement("span");
    badge.className = "class-badge " + classification;
    var labels = {
      asr: "ASR",
      reading: "READ",
      normalisation: "NORM",
      punctuation: "PUNCT",
      dialect: "DIAL",
    };
    badge.textContent = labels[classification] || classification;
    el.appendChild(badge);
  }

  function buildAnnotationsList() {
    const list = [];
    segmentsData.forEach(function (seg) {
      seg.diff_ops.forEach(function (op, opIdx) {
        var key = seg.index + "-" + opIdx;
        var ann = annotations[key] || null;
        var entry = {
          segment_index: seg.index,
          start: seg.start,
          end: seg.end,
          vibevoice: seg.vibevoice,
          etext: seg.etext,
          boilerplate: seg.boilerplate || false,
          diff_op_index: opIdx,
          op: op.op,
          vv_words: op.vv_words,
          et_words: op.et_words,
        };
        if (ann) {
          entry.classification = ann.classification;
          if (ann.normalised) {
            entry.normalised = ann.normalised;
          }
        }
        list.push(entry);
      });
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
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 0);
  }

  function doLoad() {
    var file = loadInput.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        var data = JSON.parse(e.target.result);
        if (!Array.isArray(data)) {
          setStatus("Load error: expected a JSON array");
          return;
        }
        setStatus("Loading annotations...");
        // Send to server to reconstruct segments with diff_ops
        fetch("/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        })
          .then(function (r) { return r.json(); })
          .then(function (res) {
            if (res.error) {
              setStatus("Load error: " + res.error);
              return;
            }
            segmentsData = res.segments;
            annotations = {};
            applyLoadedAnnotations(data, res.index_map || {});
            renderSegments();
            saveBtn.disabled = false;
            exportBtn.disabled = false;
            setStatus("Loaded " + data.length + " annotations across " + segmentsData.length + " segments.");
          })
          .catch(function (err) { setStatus("Load failed: " + err); });
      } catch (err) {
        setStatus("Load error: " + err);
      }
    };
    reader.readAsText(file);
  }

  function applyLoadedAnnotations(data, indexMap) {
    data.forEach(function (entry) {
      var origIdx = entry.segment_index;
      // Map original segment index to the new re-indexed position
      var segIdx = indexMap && indexMap[String(origIdx)] !== undefined
        ? indexMap[String(origIdx)]
        : origIdx;
      var seg = segmentsData[segIdx];
      if (!seg) return;

      // Find the diff_op that matches by content, not by index,
      // since recomputed diff_ops may have different indices.
      var entryVV = (entry.vv_words || []).join(" ");
      var entryET = (entry.et_words || []).join(" ");
      var matchIdx = -1;
      for (var i = 0; i < seg.diff_ops.length; i++) {
        var op = seg.diff_ops[i];
        if (op.op === entry.op &&
            (op.vv_words || []).join(" ") === entryVV &&
            (op.et_words || []).join(" ") === entryET) {
          matchIdx = i;
          break;
        }
      }
      if (matchIdx === -1) return;

      var key = segIdx + "-" + matchIdx;
      var ann = { classification: entry.classification };
      if (entry.normalised) {
        ann.normalised = entry.normalised;
      }
      annotations[key] = ann;
    });
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
