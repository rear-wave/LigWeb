"use strict";

const state = {
  config: null,
  dataset: "train",
  files: [],
  fileTotal: 0,
  fileQuery: "",
  currentFile: null,
  pieces: [],
  currentPiece: null,
  selectedPieceIndex: null,
  checked: new Set(),
  deleted: new Set(),
  zoomY: 1,
  plot: null,
  searchTimer: null,
  filesCollapsed: false,
  filesBallDrag: null,
  suppressFilesBallClick: false,
  dragDepth: 0,
  fileRequestToken: 0,
  selectionRequestToken: 0,
  saveInProgress: false,
  inferenceSignature: null,
  modelSignature: null,
  inferenceRefreshInProgress: false,
  inferenceRefreshPending: false,
  inferenceRefreshNotifyPending: false,
};

const $ = (id) => document.getElementById(id);
const REVIEW_STORAGE_KEY = "ligedit.reviewState.v1";

function loadStoredWorkspace() {
  try {
    const value = JSON.parse(localStorage.getItem(REVIEW_STORAGE_KEY) || "null");
    if (value && typeof value === "object") {
      return {
        last_open: value.last_open || null,
        files: value.files && typeof value.files === "object" ? value.files : {},
      };
    }
  } catch (_error) {
    localStorage.removeItem(REVIEW_STORAGE_KEY);
  }
  return { last_open: null, files: {} };
}

function saveStoredWorkspace(workspace) {
  try {
    localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(workspace));
  } catch (_error) {
    // The app remains usable when browser storage is disabled or full.
  }
}

function reviewStorageKey(dataset, path) {
  return `${dataset}\u0000${path}`;
}

function persistCurrentReviewState() {
  if (!state.currentFile || !state.pieces.length) return;
  const workspace = loadStoredWorkspace();
  const pieceByIndex = new Map(state.pieces.map((piece) => [piece.index, piece]));
  const eventTimes = (indices) => [...indices]
    .map((index) => pieceByIndex.get(index)?.event_time)
    .filter((value) => value !== undefined);
  workspace.last_open = { dataset: state.dataset, path: state.currentFile };
  workspace.files[reviewStorageKey(state.dataset, state.currentFile)] = {
    checked_times: eventTimes(state.checked),
    deleted_times: eventTimes(state.deleted),
    checked_indices: [...state.checked],
    deleted_indices: [...state.deleted],
    selected_time: pieceByIndex.get(state.selectedPieceIndex)?.event_time ?? null,
    selected_index: state.selectedPieceIndex,
  };
  saveStoredWorkspace(workspace);
}

function restoreReviewState(dataset, path, pieces) {
  const workspace = loadStoredWorkspace();
  const record = workspace.files[reviewStorageKey(dataset, path)] || {};
  const indexByTime = new Map(
    pieces.map((piece) => [String(piece.event_time), piece.index]),
  );
  const validIndices = new Set(pieces.map((piece) => piece.index));
  const restoreIndices = (times, indices) => {
    if (Array.isArray(times)) {
      return new Set(times
        .map((time) => indexByTime.get(String(time)))
        .filter((index) => index !== undefined));
    }
    return new Set((Array.isArray(indices) ? indices : [])
      .filter((index) => Number.isInteger(index) && validIndices.has(index)));
  };
  const selectedByTime = record.selected_time === null || record.selected_time === undefined
    ? undefined
    : indexByTime.get(String(record.selected_time));
  const selectedIndex = selectedByTime !== undefined
    ? selectedByTime
    : (validIndices.has(record.selected_index) ? record.selected_index : null);
  return {
    checked: restoreIndices(record.checked_times, record.checked_indices),
    deleted: restoreIndices(record.deleted_times, record.deleted_indices),
    selectedIndex,
  };
}

function clearStoredReviewState(dataset, path, clearLastOpen = true) {
  const workspace = loadStoredWorkspace();
  delete workspace.files[reviewStorageKey(dataset, path)];
  if (
    clearLastOpen
    && workspace.last_open?.dataset === dataset
    && workspace.last_open?.path === path
  ) {
    workspace.last_open = null;
  }
  saveStoredWorkspace(workspace);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(payload?.detail || `请求失败 (${response.status})`);
  return payload;
}

function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function toast(message, error = false) {
  const element = $("toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.remove("hidden");
  clearTimeout(element._timer);
  element._timer = setTimeout(() => element.classList.add("hidden"), 3500);
}

function setBusy(message) {
  $("pieceList").className = "piece-list empty-state";
  $("pieceList").textContent = message;
}

async function initialize() {
  bindEvents();
  try {
    state.config = await api("/api/config");
    const lastOpen = loadStoredWorkspace().last_open;
    if (state.config.datasets.some((dataset) => dataset.id === lastOpen?.dataset)) {
      state.dataset = lastOpen.dataset;
    }
    renderDatasetTabs();
    renderLabelButtons();
    setFilesCollapsed(false);
    await Promise.all([loadFiles(false), refreshHealth()]);
    if (lastOpen?.path && lastOpen.dataset === state.dataset) {
      const restored = await openFile(lastOpen.path);
      if (!restored) clearStoredReviewState(lastOpen.dataset, lastOpen.path);
    }
    setInterval(refreshHealth, 4000);
  } catch (error) {
    $("serverStatus").textContent = "连接失败";
    $("serverStatus").className = "status-pill error";
    toast(error.message, true);
  }
}

function bindEvents() {
  $("fileSearch").addEventListener("input", (event) => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => {
      state.fileQuery = event.target.value;
      loadFiles(false);
    }, 250);
  });
  $("pieceSearch").addEventListener("input", renderPieces);
  $("loadMoreButton").addEventListener("click", () => loadFiles(true));
  const collapseButton = $("collapseFilesButton");
  collapseButton.addEventListener("click", () => {
    if (state.suppressFilesBallClick) return;
    setFilesCollapsed(!state.filesCollapsed);
  });
  collapseButton.addEventListener("pointerdown", beginFilesBallDrag);
  collapseButton.addEventListener("pointermove", moveFilesBall);
  collapseButton.addEventListener("pointerup", endFilesBallDrag);
  collapseButton.addEventListener("pointercancel", endFilesBallDrag);
  $("saveLigButton").addEventListener("click", () => requestSaveLig(false));
  $("closeLigButton").addEventListener("click", closeLig);
  $("saveAndCloseButton").addEventListener("click", () => {
    hideCloseSaveDialog();
    saveCurrentLig(true);
  });
  $("discardAndCloseButton").addEventListener("click", () => {
    hideCloseSaveDialog();
    finalizeCloseLig();
  });
  $("cancelCloseButton").addEventListener("click", hideCloseSaveDialog);
  $("closeSaveDialog").addEventListener("click", (event) => {
    if (event.target === $("closeSaveDialog")) hideCloseSaveDialog();
  });
  $("importFileButton").addEventListener("click", () => $("uploadInput").click());
  $("uploadInput").addEventListener("change", (event) => {
    const files = [...event.target.files];
    event.target.value = "";
    uploadFiles(files);
  });
  $("importCorrectedButton").addEventListener("click", importCorrectedPieces);
  bindFileDrop();
  $("selectAllButton").addEventListener("click", () => {
    state.pieces.forEach((piece) => state.checked.add(piece.index));
    renderPieces();
  });
  $("clearSelectionButton").addEventListener("click", () => {
    state.checked.clear();
    renderPieces();
  });
  bindOperationMenus();
  $("cancelCorrectionButton").addEventListener("click", cancelCurrentCorrection);
  document.addEventListener("keydown", handleGlobalKeyboard);
  document.addEventListener("click", () => {
    $("contextMenu").classList.add("hidden");
    closeOperationMenus();
  });
  document.addEventListener("pointerup", clearNonEditableSelection);
  window.addEventListener("beforeunload", persistCurrentReviewState);
  const canvas = $("waveformCanvas");
  canvas.addEventListener("wheel", (event) => {
    if (!state.plot) return;
    event.preventDefault();
    state.zoomY = Math.min(20, Math.max(.25, state.zoomY * (event.deltaY < 0 ? 1.2 : .84)));
    drawPlot();
  }, { passive: false });
  canvas.addEventListener("dblclick", () => { state.zoomY = 1; drawPlot(); });
  new ResizeObserver(drawPlot).observe($("canvasWrap"));
  window.addEventListener("resize", () => {
    if (window.innerWidth <= 700 && state.filesCollapsed) {
      setFilesCollapsed(false);
    } else if (state.filesCollapsed) {
      clampFilesBallPosition();
    }
  });
}

function clearNonEditableSelection(event) {
  const target = event.target instanceof Element ? event.target : null;
  if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
  window.getSelection()?.removeAllRanges();
}

function bindOperationMenus() {
  const toggle = (buttonId, menuId, event) => {
    event.stopPropagation();
    const menu = $(menuId);
    const opening = menu.classList.contains("hidden");
    closeOperationMenus();
    menu.classList.toggle("hidden", !opening);
    $(buttonId).setAttribute("aria-expanded", String(opening));
  };
  $("exportMenuButton").addEventListener("click", (event) => {
    toggle("exportMenuButton", "exportMenu", event);
  });
  $("deleteMenuButton").addEventListener("click", (event) => {
    toggle("deleteMenuButton", "deleteMenu", event);
  });
  const action = (id, callback) => $(id).addEventListener("click", (event) => {
    event.stopPropagation();
    closeOperationMenus();
    callback();
  });
  action("exportCheckedAction", () => exportPieces("checked"));
  action("exportRemainingAction", () => exportPieces("remaining"));
  action("exportDaynightAction", exportByDaynight);
  action("exportTimestampsAction", () => exportTimestamps("remaining"));
  action("exportCheckedTimestampsAction", () => exportTimestamps("checked"));
  action("deleteCurrentAction", deleteCurrentPiece);
  action("deleteCheckedAction", deleteCheckedPieces);
  action("undoCurrentDeleteAction", undoCurrentDelete);
  action("undoAllDeleteAction", undoAllDelete);
}

function closeOperationMenus() {
  for (const id of ["exportMenu", "deleteMenu"]) $(id).classList.add("hidden");
  $("exportMenuButton").setAttribute("aria-expanded", "false");
  $("deleteMenuButton").setAttribute("aria-expanded", "false");
}

function handleGlobalKeyboard(event) {
  if (event.isComposing) return;
  if (!$("closeSaveDialog").classList.contains("hidden")) {
    if (event.key === "Escape") {
      event.preventDefault();
      hideCloseSaveDialog();
    }
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    closeOperationMenus();
    $("contextMenu").classList.add("hidden");
    cancelCurrentCheck();
    return;
  }
  if (!state.currentFile) return;
  if (["ArrowUp", "ArrowDown"].includes(event.key)) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    event.preventDefault();
    navigatePiece(event.key === "ArrowDown" ? 1 : -1);
    return;
  }
  if (event.key === "Enter" && !event.altKey && !event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    if (!event.repeat) toggleCurrentCheck();
    return;
  }
  if (["Delete", "Del"].includes(event.key) && !event.altKey && !event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    if (!event.repeat) deleteCurrentPiece();
    return;
  }
  const editable = event.target instanceof Element
    && event.target.matches("input, textarea, select, [contenteditable='true']");
  if (!editable && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoCurrentDelete();
  } else if (!editable && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "e") {
    event.preventDefault();
    exportPieces("checked");
  }
}

function visiblePieces() {
  const query = $("pieceSearch").value.trim().toLowerCase();
  return state.pieces.filter((piece) => {
    const classification = piece.classification || {};
    const searchable = `${piece.display_time} ${piece.daynight} ${classification.label || ""} ${classification.base_label || ""}`.toLowerCase();
    return !query || searchable.includes(query);
  });
}

function navigatePiece(direction) {
  const pieces = visiblePieces();
  if (!pieces.length) return;
  const current = pieces.findIndex(
    (piece) => piece.index === state.selectedPieceIndex,
  );
  const next = current < 0
    ? (direction > 0 ? 0 : pieces.length - 1)
    : Math.min(pieces.length - 1, Math.max(0, current + direction));
  if (pieces[next].index !== state.selectedPieceIndex) selectPiece(pieces[next].index);
}

function toggleCurrentCheck() {
  if (state.selectedPieceIndex === null) return toast("请先选择波形片段", true);
  const index = state.selectedPieceIndex;
  if (state.checked.has(index)) {
    state.checked.delete(index);
  } else {
    state.checked.add(index);
  }
  renderPieces();
}

function cancelCurrentCheck() {
  if (state.selectedPieceIndex === null) return;
  if (state.checked.delete(state.selectedPieceIndex)) renderPieces();
}

function setFilesCollapsed(collapsed) {
  state.filesCollapsed = Boolean(collapsed && window.innerWidth > 700);
  $("workspace").classList.toggle("files-collapsed", state.filesCollapsed);
  const button = $("collapseFilesButton");
  button.title = state.filesCollapsed ? "展开数据文件栏" : "收起数据文件栏";
  button.setAttribute("aria-label", button.title);
  button.textContent = state.filesCollapsed ? "☰" : "‹";
  if (state.filesCollapsed) requestAnimationFrame(restoreFilesBallPosition);
}

const FILES_BALL_POSITION_KEY = "ligedit.filesBallPosition";

function filesPanel() {
  return document.querySelector(".files-panel");
}

function clampFilesBallPosition(left, top) {
  const panel = filesPanel();
  if (!panel || !state.filesCollapsed) return;
  const rect = panel.getBoundingClientRect();
  const margin = 8;
  const nextLeft = Math.min(
    Math.max(margin, left ?? rect.left),
    Math.max(margin, window.innerWidth - rect.width - margin),
  );
  const nextTop = Math.min(
    Math.max(margin, top ?? rect.top),
    Math.max(margin, window.innerHeight - rect.height - margin),
  );
  panel.style.left = `${nextLeft}px`;
  panel.style.top = `${nextTop}px`;
  return { left: nextLeft, top: nextTop };
}

function restoreFilesBallPosition() {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(FILES_BALL_POSITION_KEY));
  } catch (_error) {
    localStorage.removeItem(FILES_BALL_POSITION_KEY);
  }
  if (Number.isFinite(saved?.left) && Number.isFinite(saved?.top)) {
    clampFilesBallPosition(saved.left, saved.top);
  } else {
    clampFilesBallPosition();
  }
}

function beginFilesBallDrag(event) {
  if (!state.filesCollapsed || event.button !== 0) return;
  const panel = filesPanel();
  if (!panel) return;
  const rect = panel.getBoundingClientRect();
  state.filesBallDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    left: rect.left,
    top: rect.top,
    moved: false,
  };
  event.currentTarget.setPointerCapture(event.pointerId);
  panel.classList.add("dragging");
}

function moveFilesBall(event) {
  const drag = state.filesBallDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const deltaX = event.clientX - drag.startX;
  const deltaY = event.clientY - drag.startY;
  if (!drag.moved && Math.hypot(deltaX, deltaY) < 4) return;
  drag.moved = true;
  clampFilesBallPosition(drag.left + deltaX, drag.top + deltaY);
  event.preventDefault();
}

function endFilesBallDrag(event) {
  const drag = state.filesBallDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.filesBallDrag = null;
  const panel = filesPanel();
  panel?.classList.remove("dragging");
  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
    event.currentTarget.releasePointerCapture(event.pointerId);
  }
  if (!drag.moved) return;
  const position = clampFilesBallPosition();
  if (position) {
    localStorage.setItem(FILES_BALL_POSITION_KEY, JSON.stringify(position));
  }
  state.suppressFilesBallClick = true;
  setTimeout(() => { state.suppressFilesBallClick = false; }, 0);
  event.preventDefault();
}

function bindFileDrop() {
  const zone = $("pieceDropZone");
  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  zone.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    state.dragDepth += 1;
    $("dropOverlay").classList.remove("hidden");
  });
  zone.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  zone.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    state.dragDepth = Math.max(0, state.dragDepth - 1);
    if (!state.dragDepth) $("dropOverlay").classList.add("hidden");
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    state.dragDepth = 0;
    $("dropOverlay").classList.add("hidden");
    uploadFiles([...(event.dataTransfer?.files || [])]);
  });
}

function renderDatasetTabs() {
  const container = $("datasetTabs");
  container.replaceChildren();
  state.config.datasets.forEach((dataset) => {
    const button = document.createElement("button");
    button.className = `tab${dataset.id === state.dataset ? " active" : ""}`;
    button.textContent = dataset.label;
    button.addEventListener("click", () => {
      releaseCurrentDocument();
      state.dataset = dataset.id;
      resetOpenFile(false);
      renderDatasetTabs();
      loadFiles(false);
    });
    container.append(button);
  });
}

async function loadFiles(append) {
  const offset = append ? state.files.length : 0;
  if (!append) {
    state.files = [];
    $("fileList").className = "file-list empty-state";
    $("fileList").textContent = "正在读取文件…";
  }
  try {
    const params = new URLSearchParams({
      dataset: state.dataset, query: state.fileQuery, offset: String(offset), limit: "300",
    });
    const result = await api(`/api/files?${params}`);
    state.files = append ? state.files.concat(result.files) : result.files;
    state.fileTotal = result.total;
    renderFiles();
  } catch (error) {
    $("fileList").textContent = error.message;
    toast(error.message, true);
  }
}

function renderFiles() {
  const list = $("fileList");
  list.className = "file-list";
  list.replaceChildren();
  if (!state.files.length) {
    list.className = "file-list empty-state";
    list.textContent = "没有找到 .lig 文件";
  }
  state.files.forEach((file) => {
    const button = document.createElement("button");
    button.className = `file-item${state.currentFile === file.path ? " active" : ""}`;
    button.title = file.path;
    button.innerHTML = `<span class="file-icon">ϟ</span><span class="file-copy"><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(file.path)} · ${formatBytes(file.size)}</small></span>`;
    button.addEventListener("click", () => openFile(file.path));
    list.append(button);
  });
  $("fileCount").textContent = `${state.files.length}/${state.fileTotal}`;
  $("loadMoreButton").classList.toggle("hidden", state.files.length >= state.fileTotal);
}

async function openFile(path, preferredIndex = null) {
  const switchingFile = state.currentFile !== path;
  if (state.currentFile && switchingFile) persistCurrentReviewState();
  if (state.currentFile && switchingFile) {
    releaseCurrentDocument();
  }
  const requestToken = ++state.fileRequestToken;
  state.selectionRequestToken += 1;
  state.currentFile = path;
  state.currentPiece = null;
  state.selectedPieceIndex = null;
  if (switchingFile) state.zoomY = 1;
  state.checked.clear();
  state.deleted.clear();
  $("deletedCount").textContent = "0";
  renderFiles();
  setBusy("正在解析并分类波形…");
  $("currentFileTitle").textContent = path.split("/").pop();
  $("closeLigButton").disabled = false;
  $("saveLigButton").disabled = false;
  try {
    const result = await api(`/api/files/${state.dataset}/${encodePath(path)}/pieces`);
    if (requestToken !== state.fileRequestToken || state.currentFile !== path) return;
    state.pieces = result.pieces;
    const restoredReview = restoreReviewState(state.dataset, path, state.pieces);
    state.checked = restoredReview.checked;
    state.deleted = restoredReview.deleted;
    $("pieceCount").textContent = result.piece_count;
    renderPieces();
    setFilesCollapsed(true);
    if (state.pieces.length) {
      const requestedIndex = Number.isInteger(preferredIndex)
        ? preferredIndex
        : restoredReview.selectedIndex;
      const selectedIndex = state.pieces.some((piece) => piece.index === requestedIndex)
        ? requestedIndex
        : state.pieces[0].index;
      selectPiece(selectedIndex);
    } else {
      clearDetail();
    }
    if (state.checked.size || state.deleted.size) {
      toast(`已恢复 ${state.checked.size} 个勾选和 ${state.deleted.size} 个删除标记`);
    }
    return true;
  } catch (error) {
    if (requestToken !== state.fileRequestToken) return;
    setBusy(error.message);
    toast(error.message, true);
    return false;
  }
}

function releaseCurrentDocument() {
  if (!state.currentFile) return;
  const dataset = state.dataset;
  const path = state.currentFile;
  api(`/api/files/${dataset}/${encodePath(path)}/session`, {
    method: "DELETE",
  }).catch(() => {});
}

function resetOpenFile(expandFiles = true) {
  state.fileRequestToken += 1;
  state.selectionRequestToken += 1;
  state.currentFile = null;
  state.pieces = [];
  state.currentPiece = null;
  state.selectedPieceIndex = null;
  state.checked.clear();
  state.deleted.clear();
  $("deletedCount").textContent = "0";
  state.plot = null;
  state.zoomY = 1;
  $("currentFileTitle").textContent = "波形片段";
  $("pieceCount").textContent = "0";
  $("saveLigButton").disabled = true;
  $("closeLigButton").disabled = true;
  $("pieceList").className = "piece-list empty-state";
  $("pieceList").textContent = "请选择一个 .lig 文件";
  renderFiles();
  clearDetail();
  const workspace = loadStoredWorkspace();
  workspace.last_open = null;
  saveStoredWorkspace(workspace);
  if (expandFiles) setFilesCollapsed(false);
}

async function closeLig() {
  if (!state.currentFile) return;
  if (state.deleted.size) {
    showCloseSaveDialog();
    return;
  }
  finalizeCloseLig();
}

function finalizeCloseLig() {
  if (!state.currentFile) return;
  const name = state.currentFile.split("/").pop();
  clearStoredReviewState(state.dataset, state.currentFile);
  releaseCurrentDocument();
  resetOpenFile(true);
  toast(`已关闭 ${name}`);
}

function showCloseSaveDialog() {
  const count = state.deleted.size;
  $("closeSaveMessage").textContent = `当前文件有 ${count} 个删除标记尚未保存。`;
  $("closeSaveDialog").classList.remove("hidden");
  $("saveAndCloseButton").focus();
}

function hideCloseSaveDialog() {
  $("closeSaveDialog").classList.add("hidden");
}

async function requestSaveLig(closeAfter) {
  if (!state.currentFile) return toast("请先选择文件", true);
  if (!state.deleted.size) return toast("没有需要保存的删除标记", true);
  if (!closeAfter && !window.confirm(
    `确定从原 LIG 文件中删除标记的 ${state.deleted.size} 个片段？`,
  )) return;
  await saveCurrentLig(closeAfter);
}

async function saveCurrentLig(closeAfter = false) {
  if (!state.currentFile || !state.deleted.size || state.saveInProgress) return;
  const dataset = state.dataset;
  const path = state.currentFile;
  const deleted = new Set(state.deleted);
  const remainingOldIndices = state.pieces
    .map((piece) => piece.index)
    .filter((index) => !deleted.has(index));
  let preferredOldIndex = state.selectedPieceIndex;
  if (preferredOldIndex === null || deleted.has(preferredOldIndex)) {
    preferredOldIndex = remainingOldIndices.find(
      (index) => state.selectedPieceIndex === null || index > state.selectedPieceIndex,
    ) ?? remainingOldIndices.at(-1) ?? null;
  }
  const preferredNewIndex = preferredOldIndex === null
    ? null
    : remainingOldIndices.indexOf(preferredOldIndex);
  state.saveInProgress = true;
  $("saveLigButton").disabled = true;
  $("closeLigButton").disabled = true;
  try {
    const result = await api(`/api/files/${dataset}/${encodePath(path)}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ deleted_indices: [...deleted] }),
    });
    clearStoredReviewState(dataset, path, false);
    state.deleted.clear();
    toast(`已保存：删除 ${result.deleted_count} 条，保留 ${result.piece_count} 条`);
    if (closeAfter) {
      finalizeCloseLig();
    } else {
      await openFile(path, preferredNewIndex);
      await loadFiles(false);
    }
  } catch (error) {
    toast(`保存失败：${error.message}`, true);
  } finally {
    state.saveInProgress = false;
    if (state.currentFile) {
      $("closeLigButton").disabled = false;
      $("saveLigButton").disabled = false;
    }
  }
}

function renderPieces() {
  const pieces = visiblePieces();
  const list = $("pieceList");
  $("deletedCount").textContent = String(state.deleted.size);
  $("saveLigButton").disabled = !state.currentFile || state.saveInProgress;
  persistCurrentReviewState();
  list.className = "piece-list";
  list.replaceChildren();
  if (!pieces.length) {
    list.className = "piece-list empty-state";
    list.textContent = state.currentFile ? "没有匹配的片段" : "请选择一个 .lig 文件";
    return;
  }
  pieces.forEach((piece) => {
    const row = document.createElement("div");
    const correction = piece.classification?.correction_model || {};
    row.className = `piece-row${state.selectedPieceIndex === piece.index ? " active" : ""}${state.deleted.has(piece.index) ? " deleted" : ""}`;
    row.dataset.pieceIndex = String(piece.index);
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.checked.has(piece.index);
    checkbox.addEventListener("click", (event) => {
      event.stopPropagation();
      checkbox.checked ? state.checked.add(piece.index) : state.checked.delete(piece.index);
      persistCurrentReviewState();
    });
    const copy = document.createElement("div");
    copy.className = "piece-time";
    copy.innerHTML = `<strong>${escapeHtml(piece.display_time)}</strong><small>#${piece.index + 1} · ${piece.daynight} · ${piece.sample_count} 点</small>`;
    const tag = document.createElement("span");
    const mainLabel = piece.classification?.main_model?.label || piece.classification?.base_label || "—";
    const correctedLabel = piece.classification?.label || correction.label || "—";
    tag.className = `class-tag${mainLabel !== correctedLabel ? " differing" : ""}`;
    tag.textContent = correctedLabel;
    tag.title = `纠错后结果：${correctedLabel}`;
    row.append(checkbox, copy, tag);
    row.addEventListener("click", () => selectPiece(piece.index));
    row.addEventListener("dblclick", () => {
      state.checked.has(piece.index) ? state.checked.delete(piece.index) : state.checked.add(piece.index);
      renderPieces();
    });
    row.addEventListener("contextmenu", (event) => showContextMenu(event, piece));
    list.append(row);
  });
  requestAnimationFrame(() => {
    const active = list.querySelector(".piece-row.active");
    active?.scrollIntoView({ block: "nearest" });
  });
}

async function selectPiece(index) {
  if (!state.currentFile) return;
  state.selectedPieceIndex = index;
  renderPieces();
  const requestToken = ++state.selectionRequestToken;
  const selectedFile = state.currentFile;
  const selectedDataset = state.dataset;
  try {
    const detail = await api(`/api/files/${selectedDataset}/${encodePath(selectedFile)}/piece/${index}`);
    if (
      requestToken !== state.selectionRequestToken
      || state.currentFile !== selectedFile
      || state.dataset !== selectedDataset
    ) return;
    state.currentPiece = detail;
    renderPieces();
    renderDetail();
  } catch (error) {
    if (requestToken !== state.selectionRequestToken) return;
    toast(error.message, true);
  }
}

function renderDetail() {
  const piece = state.currentPiece;
  if (!piece) return clearDetail();
  $("pieceTitle").textContent = `${piece.display_time} · #${piece.index + 1}`;
  $("pieceSubtitle").textContent = `${state.currentFile} · ${piece.daynight}`;
  const classification = piece.classification;
  const main = classification.main_model || { label: classification.base_label, confidence: classification.confidence, probabilities: classification.probabilities };
  const correction = classification.correction_model || { label: classification.label, source: classification.source, similarity: classification.similarity };
  $("mainModelLabel").textContent = main.label;
  $("mainModelMeta").textContent = `置信度 ${(main.confidence * 100).toFixed(1)}%`;
  $("correctionModelLabel").textContent = correction.label;
  const sourceNames = { base: "基础模型", manual_exact: "人工纠正", adapter: "纠错模型", dataset_label: "纠错集标签" };
  const similarity = correction.similarity === null || correction.similarity === undefined ? "" : ` · 相似度 ${(correction.similarity * 100).toFixed(1)}%`;
  $("correctionModelMeta").textContent = `${sourceNames[correction.source] || correction.source}${similarity}`;
  $("sourceBadge").textContent = sourceNames[classification.source] || classification.source;
  $("sourceBadge").style.background = classification.source === "base" ? "#edf1f3" : "#fce8e3";
  $("sourceBadge").style.color = classification.source === "base" ? "#617078" : "#a73520";
  renderProbabilities(main.probabilities);
  $("cancelCorrectionButton").classList.toggle("hidden", classification.source !== "manual_exact");
  const metadata = { "发生时间": piece.display_time, "昼夜": piece.daynight, "原始点数": piece.waveform.original_sample_count, ...piece.metadata };
  const list = $("metadataList");
  list.replaceChildren();
  Object.entries(metadata).forEach(([key, value]) => {
    const term = document.createElement("dt"); term.textContent = metadataLabel(key);
    const detailElement = document.createElement("dd"); detailElement.textContent = value ?? "—";
    list.append(term, detailElement);
  });
  state.plot = piece.waveform;
  $("plotEmpty").classList.add("hidden");
  drawPlot();
}

function renderProbabilities(probabilities) {
  const container = $("probabilityBars");
  container.replaceChildren();
  Object.entries(probabilities).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `<strong>${label}</strong><span class="bar-track"><span class="bar-value" style="width:${Math.max(0, Math.min(100, value * 100))}%"></span></span><span>${(value * 100).toFixed(1)}%</span>`;
    container.append(row);
  });
}

function renderLabelButtons() {
  const container = $("labelButtons");
  container.replaceChildren();
  state.config.classes.forEach((label) => {
    const button = document.createElement("button");
    button.className = "label-button";
    button.textContent = label;
    button.addEventListener("click", () => correctPiece(state.currentPiece?.index, label));
    container.append(button);
  });
}

async function correctPiece(index, label) {
  if (index === null || index === undefined || !state.currentFile) return;
  try {
    await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: state.dataset, path: state.currentFile, piece_index: index, corrected_label: label }),
    });
    const summary = state.pieces.find((piece) => piece.index === index);
    if (summary?.classification) {
      summary.classification.label = label;
      summary.classification.source = "manual_exact";
      summary.classification.correction_model = {
        ...(summary.classification.correction_model || {}),
        label,
        applied: true,
        source: "manual_exact",
        similarity: null,
      };
    }
    toast(`已记录：#${index + 1} → ${label}`);
    await selectPiece(index);
    refreshHealth();
  } catch (error) { toast(error.message, true); }
}

async function cancelCurrentCorrection() {
  if (!state.currentPiece) return;
  const pieceIndex = state.currentPiece.index;
  try {
    await api(`/api/feedback/${state.currentPiece.waveform_hash}`, { method: "DELETE" });
    toast("已取消本条人工纠错");
    await openFile(state.currentFile);
    await selectPiece(pieceIndex);
  } catch (error) { toast(error.message, true); }
}

function showContextMenu(event, piece) {
  event.preventDefault();
  closeOperationMenus();
  const menu = $("contextMenu");
  menu.replaceChildren();
  const preview = menuButton("预览波形", () => selectPiece(piece.index));
  const check = menuButton(state.checked.has(piece.index) ? "取消勾选" : "勾选", () => {
    state.checked.has(piece.index) ? state.checked.delete(piece.index) : state.checked.add(piece.index);
    renderPieces();
  });
  const remove = menuButton(state.deleted.has(piece.index) ? "撤销删除标记" : "标记删除", () => {
    state.deleted.has(piece.index) ? state.deleted.delete(piece.index) : state.deleted.add(piece.index);
    renderPieces();
  });
  menu.append(preview, check, remove, document.createElement("hr"));
  state.config.classes.forEach((label) => menu.append(menuButton(`纠正为 ${label}`, () => correctPiece(piece.index, label))));
  menu.style.left = `${Math.min(event.clientX, window.innerWidth - 190)}px`;
  menu.style.top = `${Math.min(event.clientY, window.innerHeight - 260)}px`;
  menu.classList.remove("hidden");
}

function menuButton(label, callback) {
  const button = document.createElement("button");
  button.textContent = label;
  button.addEventListener("click", callback);
  return button;
}

async function refreshHealth() {
  try {
    const result = await api("/api/health");
    $("serverStatus").textContent = "服务器在线";
    $("serverStatus").className = "status-pill online";
    updateTrainingStatus(result.training);
    const signature = inferenceSignature(result.training);
    const modelVersion = modelSignature(result.training);
    const changed = state.inferenceSignature !== null
      && state.inferenceSignature !== signature;
    const modelChanged = state.modelSignature !== null
      && state.modelSignature !== modelVersion;
    state.inferenceSignature = signature;
    state.modelSignature = modelVersion;
    if (changed) refreshCurrentInference(modelChanged);
  } catch (_error) {
    $("serverStatus").textContent = "服务器离线";
    $("serverStatus").className = "status-pill error";
  }
}

function inferenceSignature(training) {
  const correction = training.correction || training;
  const main = training.main || {};
  return JSON.stringify([
    main.model_hash || "",
    Number(correction.generation || 0),
    Number(correction.inference_revision || 0),
  ]);
}

function modelSignature(training) {
  const correction = training.correction || training;
  const main = training.main || {};
  return JSON.stringify([
    main.model_hash || "",
    Number(correction.generation || 0),
  ]);
}

async function refreshCurrentInference(notifyModelChange = false) {
  if (!state.currentFile) return;
  if (state.inferenceRefreshInProgress) {
    state.inferenceRefreshPending = true;
    state.inferenceRefreshNotifyPending ||= notifyModelChange;
    return;
  }
  state.inferenceRefreshInProgress = true;
  const dataset = state.dataset;
  const path = state.currentFile;
  const previousIndex = state.selectedPieceIndex;
  persistCurrentReviewState();
  try {
    const result = await api(`/api/files/${dataset}/${encodePath(path)}/pieces`);
    if (state.dataset !== dataset || state.currentFile !== path) return;
    state.pieces = result.pieces;
    const restoredReview = restoreReviewState(dataset, path, state.pieces);
    state.checked = restoredReview.checked;
    state.deleted = restoredReview.deleted;
    const selectedIndex = restoredReview.selectedIndex
      ?? (state.pieces.some((piece) => piece.index === previousIndex)
        ? previousIndex
        : state.pieces[0]?.index ?? null);
    state.selectedPieceIndex = selectedIndex;
    $("pieceCount").textContent = result.piece_count;
    renderPieces();
    if (selectedIndex !== null) await selectPiece(selectedIndex);
    if (notifyModelChange) {
      toast("模型已更新，当前文件的推理结果已同步");
    }
  } catch (error) {
    if (state.dataset === dataset && state.currentFile === path) {
      toast(`推理结果刷新失败：${error.message}`, true);
    }
  } finally {
    state.inferenceRefreshInProgress = false;
    if (state.inferenceRefreshPending) {
      const notifyPending = state.inferenceRefreshNotifyPending;
      state.inferenceRefreshPending = false;
      state.inferenceRefreshNotifyPending = false;
      refreshCurrentInference(notifyPending);
    }
  }
}

function updateTrainingStatus(training) {
  const labels = { activated: "模型已激活", pending: "有待训练纠错", queued: "训练已排队", retained: "保留现有模型", failed: "训练失败", no_changes: "模型已是最新", idle: "尚未训练" };
  const correction = training.correction || training;
  const main = training.main || {};
  const icSync = training.ic_sync || {};
  const icLabels = {
    synced: "已同步",
    waiting: "等待同步",
    failed: "同步失败",
    disabled: "已关闭",
  };
  $("icSyncStatus").textContent = icSync.running
    ? "IC 同步：进行中…"
    : `IC 同步：${icLabels[icSync.status] || icSync.status || "等待同步"}${Number.isInteger(icSync.synced_pieces) ? ` · ${icSync.synced_pieces} 条` : ""}`;
  $("icSyncStatus").title = icSync.reason || "纠错集 IC 自动同步到训练集";
  $("trainingStatus").textContent = correction.running ? "纠错模型：训练中…" : `纠错模型：${labels[correction.status] || correction.status} · ${correction.record_count} 条 · G${correction.generation}`;
  const mainLabels = { activated: "已激活", preparing: "准备数据", training: "训练中", exporting: "导出中", failed: "失败", waiting: "等待 22:00" };
  $("mainTrainingStatus").textContent = `主模型：${mainLabels[main.status] || main.status || "等待 22:00"}`;
  $("mainTrainingStatus").title = `${main.reason || "每天 22:00 自动训练"}${training.automation?.next_main_training ? `；下次 ${new Date(training.automation.next_main_training).toLocaleString()}` : ""}`;
}

async function exportPieces(mode) {
  if (!state.currentFile) return toast("请先选择文件", true);
  const keep = mode === "checked"
    ? [...state.checked]
    : state.pieces.map((piece) => piece.index).filter((index) => !state.deleted.has(index));
  if (!keep.length) return toast("没有可导出的片段", true);
  const base = state.currentFile.split("/").pop().replace(/\.lig$/i, "");
  const outputName = `${base}_${mode === "checked" ? "selected" : "remaining"}.lig`;
  try {
    const result = await api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: state.dataset, path: state.currentFile, keep_indices: keep, output_name: outputName }),
    });
    toast(`已导出 ${result.piece_count} 条波形`);
    window.location.assign(result.download_url);
  } catch (error) { toast(error.message, true); }
}

async function exportByDaynight() {
  if (!state.currentFile) return toast("请先选择文件", true);
  const remaining = state.pieces.length - state.deleted.size;
  if (!remaining) return toast("没有未删除的片段", true);
  const base = state.currentFile.split("/").pop().replace(/\.lig$/i, "");
  try {
    const result = await api("/api/export/daynight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: state.dataset,
        path: state.currentFile,
        excluded_indices: [...state.deleted],
        output_name: `${base}_daynight.zip`,
      }),
    });
    toast(`已按昼夜导出：白天 ${result.day_count} 条，夜晚 ${result.night_count} 条`);
    window.location.assign(result.download_url);
  } catch (error) { toast(error.message, true); }
}

function exportTimestamps(mode) {
  if (!state.currentFile) return toast("请先选择文件", true);
  const pieces = mode === "checked"
    ? state.pieces.filter((piece) => state.checked.has(piece.index))
    : state.pieces.filter((piece) => !state.deleted.has(piece.index));
  if (!pieces.length) return toast("没有可导出的时间戳", true);
  const base = state.currentFile.split("/").pop().replace(/\.lig$/i, "");
  const suffix = mode === "checked" ? "selected_timestamps" : "timestamps";
  downloadText(`${base}_${suffix}.txt`, pieces.map((piece) => piece.event_time).join("\n") + "\n");
  toast(`已导出 ${pieces.length} 条时间戳`);
}

function downloadText(filename, content) {
  const url = URL.createObjectURL(new Blob(["\ufeff", content], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function deleteCurrentPiece() {
  if (state.selectedPieceIndex === null) return toast("请先选择波形片段", true);
  state.deleted.add(state.selectedPieceIndex);
  renderPieces();
  toast(`已标记删除第 ${state.selectedPieceIndex + 1} 条`);
}

function deleteCheckedPieces() {
  if (!state.checked.size) return toast("没有勾选的片段", true);
  let added = 0;
  for (const index of state.checked) {
    if (!state.deleted.has(index)) added += 1;
    state.deleted.add(index);
  }
  renderPieces();
  toast(added ? `已标记删除 ${added} 条勾选片段` : "勾选片段均已标记删除");
}

function undoCurrentDelete() {
  if (state.selectedPieceIndex === null) return toast("请先选择波形片段", true);
  if (!state.deleted.delete(state.selectedPieceIndex)) {
    return toast("当前片段没有删除标记", true);
  }
  renderPieces();
  toast(`已撤销第 ${state.selectedPieceIndex + 1} 条的删除标记`);
}

function undoAllDelete() {
  if (!state.deleted.size) return toast("没有标记删除的片段", true);
  const count = state.deleted.size;
  if (!window.confirm(`撤销全部 ${count} 个删除标记？`)) return;
  state.deleted.clear();
  renderPieces();
  toast(`已撤销 ${count} 个删除标记`);
}

async function importCorrectedPieces() {
  if (!state.currentFile) return toast("请先选择已检查完成的 LIG 文件", true);
  const processedDataset = state.dataset;
  const processedPath = state.currentFile;
  const indices = state.pieces
    .map((piece) => piece.index)
    .filter((index) => !state.deleted.has(index));
  if (!indices.length) return toast("当前 LIG 没有可加入纠错数据集的片段", true);
  try {
    $("importCorrectedButton").disabled = true;
    const result = await api("/api/correction-imports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: processedDataset,
        path: processedPath,
        piece_indices: indices,
      }),
    });
    let message;
    if (!result.imported_piece_count) {
      const duplicate = result.duplicate_skipped_count
        ? `已存在波形 ${result.duplicate_skipped_count} 条已去重`
        : "";
      message = `检查完成，没有新增片段${duplicate ? `：${duplicate}` : ""}`;
    } else {
      const labels = result.files.map((item) => `${item.label} ${item.piece_count} 条`).join("，");
      const sources = `；人工结果 ${result.manual_piece_count} 条，纠错模型结果 ${result.model_piece_count} 条`;
      const duplicate = result.duplicate_skipped_count
        ? `；已跳过 ${result.duplicate_skipped_count} 条重复波形`
        : "";
      const reclassified = result.reclassified_piece_count
        ? `；${result.reclassified_piece_count} 条已有波形已按人工结果重新归类`
        : "";
      const deleted = state.deleted.size ? `；排除 ${state.deleted.size} 条删除标记` : "";
      message = `检查完成并加入纠错数据集：${labels}${sources}${duplicate}${reclassified}${deleted}`;
    }
    if (result.source_removed) {
      clearStoredReviewState(processedDataset, processedPath);
      resetOpenFile(true);
      await loadFiles(false);
      message += "；原 LIG 已从待处理数据集中删除";
    } else if (state.dataset === "correction") {
      await loadFiles(false);
    }
    toast(message);
  } catch (error) { toast(error.message, true); }
  finally { $("importCorrectedButton").disabled = false; }
}

async function uploadFiles(files) {
  const ligFiles = files.filter((file) => file.name.toLowerCase().endsWith(".lig"));
  if (!ligFiles.length) return toast("请拖入 .lig 文件", true);
  if (ligFiles.length !== files.length) toast("已忽略非 .lig 文件");
  const imported = [];
  try {
    $("importFileButton").disabled = true;
    for (let index = 0; index < ligFiles.length; index += 1) {
      const file = ligFiles[index];
      $("importFileButton").textContent = `上传 ${index + 1}/${ligFiles.length}`;
      const result = await api(`/api/uploads?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
      imported.push(result);
    }
    toast(`已上传 ${imported.length} 个 .lig 文件到待处理区，人工确认后再添加到纠错集`);
    releaseCurrentDocument();
    state.dataset = "inbox";
    resetOpenFile(false);
    state.fileQuery = "";
    $("fileSearch").value = "";
    renderDatasetTabs();
    await loadFiles(false);
    if (imported.length) await openFile(imported[imported.length - 1].path);
  } catch (error) {
    toast(`已完成 ${imported.length} 个；${error.message}`, true);
  } finally {
    $("importFileButton").disabled = false;
    $("importFileButton").textContent = "上传待处理 LIG";
  }
}

function drawPlot() {
  if (!state.plot) return;
  const canvas = $("waveformCanvas");
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  const margin = { left: 50, right: 17, top: 15, bottom: 32 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#e3e8ea"; context.lineWidth = 1;
  context.fillStyle = "#718088"; context.font = "10px Segoe UI";
  for (let i = 0; i <= 5; i++) {
    const x = margin.left + plotWidth * i / 5;
    context.beginPath(); context.moveTo(x, margin.top); context.lineTo(x, margin.top + plotHeight); context.stroke();
    const value = state.plot.time_ms[Math.round((state.plot.time_ms.length - 1) * i / 5)] || 0;
    context.fillText(`${value.toFixed(2)} ms`, x - 14, height - 10);
  }
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + plotHeight * i / 4;
    context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
  }
  const all = state.plot.raw.concat(state.plot.filtered);
  let extent = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)), 1) / state.zoomY;
  const mapX = (index) => margin.left + plotWidth * index / Math.max(1, state.plot.raw.length - 1);
  const mapY = (value) => margin.top + plotHeight / 2 - value / extent * plotHeight * .46;
  drawSeries(context, state.plot.raw, mapX, mapY, "#9aa5aa", .8);
  drawSeries(context, state.plot.filtered, mapX, mapY, "#e45435", 1.35);
  context.fillStyle = "#718088";
  context.fillText(`±${extent.toFixed(1)}`, 5, margin.top + 4);
  context.fillText("0", 30, margin.top + plotHeight / 2 + 3);
}

function drawSeries(context, values, mapX, mapY, color, lineWidth) {
  context.beginPath(); context.strokeStyle = color; context.lineWidth = lineWidth;
  values.forEach((value, index) => {
    const x = mapX(index), y = mapY(value);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

function clearDetail() {
  state.plot = null;
  $("pieceTitle").textContent = "请选择波形";
  $("pieceSubtitle").textContent = "从左侧选择服务器上的数据文件";
  $("mainModelLabel").textContent = "—";
  $("mainModelMeta").textContent = "尚未运行";
  $("correctionModelLabel").textContent = "—";
  $("correctionModelMeta").textContent = "尚未运行";
  $("probabilityBars").replaceChildren();
  $("plotEmpty").classList.remove("hidden");
  const canvas = $("waveformCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

function metadataLabel(key) {
  const labels = { m_samplingRate: "采样率", m_numOfData: "数据点", m_numOfChannel: "通道", m_stationID: "站点 ID", m_stationName: "站点名称", m_GPSCurrentLocationLat: "纬度", m_GPSCurrentLocationLon: "经度", m_preTriggerNum: "预触发点", m_Range: "量程", version: "版本" };
  return labels[key] || key;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

initialize();
