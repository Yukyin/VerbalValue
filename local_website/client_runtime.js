// VerbalValue client runtime (open-source reference implementation).
//
// Demonstrates the "Client Runtime" and "Sentence-level Streaming TTS"
// components from the paper's architecture (Figure 2):
//
//   - splitByPunctuation / speakSegments implement clause segmentation
//     and incremental, interruptible sentence-level TTS playback.
//   - The idle pitch channel (pollIdleNext) and the interactive response
//     channel (sendComment) arbitrate a single shared audio resource via
//     a small set of state flags (idleBusy, idlePaused, suspendIdleUntil,
//     replyBusy), implementing the dual-channel session arbitration
//     described in Section 4.2: on comment arrival the interactive
//     channel preempts the idle channel, plays the response, and then
//     resumes idle narration from the saved sentence boundary after a
//     configurable hold period.
//
// This file omits UI styling, layout, and any product- or deployment-
// specific configuration. Endpoint URLs and timing constants are left
// as configuration placeholders (see CONFIG below).

const CONFIG = {
  // Base URL of the dialogue service (POST /chat, GET /idle_next).
  CHAT_BASE: "",
  // Base URL of the media service (POST /speak).
  SPEAK_BASE: "",
  // Idle-channel polling interval, in milliseconds.
  IDLE_POLL_MS: null,
  // How long to hold the audio resource for the interactive channel
  // before resuming idle narration, in milliseconds.
  RESUME_IDLE_AFTER_MS: null,
  // Short freeze window applied immediately after a reply arrives, to
  // avoid a race with the idle poll loop, in milliseconds.
  REPLY_SUSPEND_IDLE_MS: null,
  // Polling interval used while waiting for the current audio segment
  // to finish or be aborted, in milliseconds.
  ABORT_WATCHER_POLL_MS: null,
};

// ---------------------------------------------------------------------------
// Session arbitration state
// ---------------------------------------------------------------------------
//
// These flags implement the dual-channel arbitration conceptually
// described as a session state machine (Idle / PlayingIdle /
// AwaitingReply / PlayingReply / Resuming):
//   - idleBusy / idleLoopRunning: idle channel poll loop bookkeeping
//   - idlePaused: idle channel suspended while a reply is in flight
//   - replyBusy: interactive channel currently playing a response
//   - suspendIdleUntil: short freeze window to avoid race conditions
//     when handing the audio resource back to the idle channel
//   - idleState: saved {product_id, segments, idx} so idle narration
//     can resume from the exact sentence boundary it was interrupted at

let pendingChat = false;
let queuedComment = null;
let isSpeaking = false;
let replyBusy = false;
let speakAbort = { flag: false };
let idleBusy = false;
let idleState = null; // { product_id, segments, idx }
let suspendIdleUntil = 0;
let idleLoopRunning = false;
let idlePaused = false;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function suspendIdle(ms) {
  suspendIdleUntil = Date.now() + Math.max(0, Number(ms || 0));
}

function clearSuspendIdle() {
  suspendIdleUntil = 0;
}

// ---------------------------------------------------------------------------
// Clause segmentation
// ---------------------------------------------------------------------------

/**
 * Split spoken text into clauses on sentence-ending punctuation, so each
 * clause can be synthesised and played independently. This is the
 * "Clause segmentation" step shown in Figure 2's Media Service.
 */
function splitByPunctuation(text) {
  const clean = (text || "").trim();
  if (!clean) return [];
  const segs = clean.match(/[^.!?\u3002\uff01\uff1f]+[.!?\u3002\uff01\uff1f]?/g);
  if (!segs) return [clean];
  return segs.map((s) => s.trim()).filter(Boolean);
}

// ---------------------------------------------------------------------------
// Streaming TTS playback
// ---------------------------------------------------------------------------

/**
 * Play a sequence of text segments by requesting synthesised audio for
 * each one in turn and playing it before requesting the next. Supports
 * resuming from an arbitrary segment index (startIndex) and reporting
 * progress via onProgress, which idleState uses to save its place when
 * interrupted.
 */
async function speakSegments(audioEl, segments, opts = {}) {
  if (!segments || segments.length === 0) return;

  const myAbort = { flag: false };
  speakAbort = myAbort;

  const startIndex = Number.isFinite(opts.startIndex) ? opts.startIndex : 0;
  const onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;

  isSpeaking = true;

  try {
    for (let i = startIndex; i < segments.length; i++) {
      if (myAbort.flag) break;

      const seg = (segments[i] || "").trim();
      if (!seg) {
        if (onProgress) onProgress(i + 1);
        continue;
      }

      if (onProgress) onProgress(i);

      const resp = await fetch(CONFIG.SPEAK_BASE + "/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: seg }),
      });

      if (myAbort.flag) break;
      if (!resp.ok) break;

      const data = await resp.json();
      if (!data.audio_wav_base64) break;

      audioEl.src = "data:audio/wav;base64," + data.audio_wav_base64;

      await new Promise((resolve, reject) => {
        const abortWatcher = setInterval(() => {
          if (myAbort.flag) {
            clearInterval(abortWatcher);
            try {
              audioEl.pause();
              audioEl.currentTime = 0;
            } catch (e) {}
            resolve();
          }
        }, CONFIG.ABORT_WATCHER_POLL_MS || 0);

        audioEl.onended = () => {
          clearInterval(abortWatcher);
          resolve();
        };
        audioEl.onerror = (e) => {
          clearInterval(abortWatcher);
          reject(e || new Error("audio error"));
        };
        audioEl.play().catch((err) => {
          clearInterval(abortWatcher);
          reject(err);
        });
      });

      if (myAbort.flag) break;
      if (onProgress) onProgress(i + 1);
    }
  } finally {
    isSpeaking = false;
  }
}

function abortSpeakingNow(audioEl) {
  try {
    speakAbort.flag = true;
    if (audioEl && !audioEl.paused) {
      audioEl.pause();
      audioEl.currentTime = 0;
    }
  } catch (e) {}
}

// ---------------------------------------------------------------------------
// Idle pitch channel
// ---------------------------------------------------------------------------

/**
 * Poll the idle channel for the next pitch-script segment and play it,
 * resuming from a saved sentence boundary if a previous idle item was
 * interrupted. Guarded by the arbitration flags so it never plays while
 * the interactive channel holds the audio resource.
 */
async function pollIdleNext(audioEl, hooks = {}) {
  if (idleBusy) return;
  if (idlePaused) return;
  if (Date.now() < suspendIdleUntil) return;
  if (replyBusy) return;
  if (isSpeaking) return;
  if (audioEl && !audioEl.paused) return;

  idleBusy = true;
  try {
    if (idleState && Array.isArray(idleState.segments) && idleState.idx < idleState.segments.length) {
      if (hooks.onProductSwitch) hooks.onProductSwitch(idleState.product_id, "resume");
      await speakSegments(audioEl, idleState.segments, {
        startIndex: idleState.idx,
        onProgress: (k) => {
          if (idleState) idleState.idx = k;
        },
      });
      if (idleState && idleState.idx >= idleState.segments.length) idleState = null;
      return;
    }

    const resp = await fetch(CONFIG.CHAT_BASE + "/idle_next", { cache: "no-store" });
    const data = await resp.json();

    if (data && data.ready && data.content) {
      const pid = (data.product_id || "").trim();
      const content = String(data.content || "").trim();

      if (pid && hooks.onProductSwitch) hooks.onProductSwitch(pid, "idle");

      const segs = splitByPunctuation(content);
      idleState = { product_id: pid, segments: segs, idx: 0 };

      await speakSegments(audioEl, segs, {
        startIndex: 0,
        onProgress: (k) => {
          if (idleState) idleState.idx = k;
        },
      });

      if (idleState && idleState.idx >= idleState.segments.length) idleState = null;
    }
  } catch (e) {
    if (hooks.onError) hooks.onError(e);
  } finally {
    idleBusy = false;
  }
}

function startIdlePolling(audioEl, hooks = {}) {
  if (idleLoopRunning) return;
  idleLoopRunning = true;
  (async () => {
    while (idleLoopRunning) {
      try {
        await pollIdleNext(audioEl, hooks);
      } catch (e) {
        if (hooks.onError) hooks.onError(e);
      }
      await sleep(CONFIG.IDLE_POLL_MS || 0);
    }
  })();
}

function stopIdlePolling() {
  idleLoopRunning = false;
}

// ---------------------------------------------------------------------------
// Interactive response channel
// ---------------------------------------------------------------------------

/**
 * Send a viewer comment to the interactive channel, preempt the idle
 * channel for the duration of the reply, play the response, and then
 * resume idle narration after CONFIG.RESUME_IDLE_AFTER_MS.
 */
async function sendComment(audioEl, text, hooks = {}) {
  const comment = (text || "").trim();
  if (!comment) return;

  if (pendingChat) {
    queuedComment = comment;
    return;
  }

  pendingChat = true;

  try {
    const resp = await fetch(CONFIG.CHAT_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ danmu: comment }),
    });
    const data = await resp.json();

    replyBusy = true;
    idlePaused = true;
    suspendIdle(CONFIG.REPLY_SUSPEND_IDLE_MS || 0);
    abortSpeakingNow(audioEl);
    isSpeaking = true;
    pendingChat = false;

    let spoken = "";
    if (typeof data.spoken === "string" && data.spoken.trim()) {
      spoken = data.spoken.trim();
    } else if (Array.isArray(data.speak_lines) && data.speak_lines.length > 0) {
      spoken = data.speak_lines.join(" ").trim();
    }

    const pid = String(data.product_id || "").trim();
    if (pid && hooks.onProductSwitch) hooks.onProductSwitch(pid, "interactive");

    if (hooks.onReply) hooks.onReply(comment, spoken);

    if (spoken) {
      const segments = splitByPunctuation(spoken);
      await speakSegments(audioEl, segments);
    } else {
      isSpeaking = false;
    }

    replyBusy = false;
    clearSuspendIdle();
    suspendIdle(CONFIG.RESUME_IDLE_AFTER_MS || 0);
    setTimeout(() => {
      clearSuspendIdle();
      idlePaused = false;
      pollIdleNext(audioEl, hooks);
    }, CONFIG.RESUME_IDLE_AFTER_MS || 0);

    if (queuedComment) {
      const q = queuedComment;
      queuedComment = null;
      sendComment(audioEl, q, hooks);
    }
  } catch (e) {
    pendingChat = false;
    isSpeaking = false;
    replyBusy = false;
    idlePaused = false;
    clearSuspendIdle();
    suspendIdle(CONFIG.RESUME_IDLE_AFTER_MS || 0);
    setTimeout(() => {
      idlePaused = false;
      pollIdleNext(audioEl, hooks);
    }, CONFIG.RESUME_IDLE_AFTER_MS || 0);
    if (hooks.onError) hooks.onError(e);
  }
}

export {
  CONFIG,
  splitByPunctuation,
  speakSegments,
  abortSpeakingNow,
  pollIdleNext,
  startIdlePolling,
  stopIdlePolling,
  sendComment,
};
