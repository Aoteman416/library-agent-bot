/* 图书馆智能客服 - 前端逻辑（原生 JS + SSE 流式） */
(function () {
  "use strict";

  const DOM = {
    list: document.getElementById("messageList"),
    input: document.getElementById("inputBox"),
    send: document.getElementById("sendBtn"),
    newChat: document.getElementById("newChatBtn"),
    login: document.getElementById("loginBtn"),
    exit: document.getElementById("exitBtn"),
    mode: document.getElementById("modeBadge"),
    actions: document.getElementById("quickActions"),
  };

  // ============ 会话状态 ============
  let sessionId = localStorage.getItem("lib_session_id") || null;
  let userId = localStorage.getItem("lib_user_id") || "";
  let streaming = false;

  // ============ 工具函数 ============
  function esc(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function scrollToBottom() {
    DOM.list.scrollTop = DOM.list.scrollHeight;
  }

  function userDisplayName() {
    return userId ? `学号 ${userId}` : "游客";
  }

  // ============ 消息渲染 ============
  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    wrap.innerHTML =
      `<span class="meta">${esc(userDisplayName())}</span>` +
      `<div class="bubble">${esc(text)}</div>`;
    DOM.list.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addAssistantMessage(metaText) {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.innerHTML =
      `<span class="meta">${esc(metaText || "智能客服")}</span>` +
      `<div class="bubble streaming"></div>`;
    DOM.list.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function renderCard(wrap, extra) {
    if (!extra) return;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = Object.entries(extra)
      .map(([k, v]) => `<div class="card-row"><span class="k">${esc(k)}</span><span class="v">${esc(String(v))}</span></div>`)
      .join("");
    wrap.appendChild(card);
    scrollToBottom();
  }

  function renderSources(wrap, documents) {
    if (!documents || !documents.length) return;
    const box = document.createElement("details");
    box.className = "sources";
    box.innerHTML =
      `<summary>📎 参考来源（${documents.length}）</summary>` +
      `<ul>${documents.map((d) => `<li>${esc(d.title)}（${esc(d.category)}）· 相关度 ${esc(d.score)}</li>`).join("")}</ul>`;
    wrap.appendChild(box);
    scrollToBottom();
  }

  function renderTransfer(wrap) {
    const tip = document.createElement("div");
    tip.className = "transfer-tip";
    tip.textContent = "⚠️ 您将被转接至人工客服，请耐心等待。";
    wrap.appendChild(tip);
    scrollToBottom();
  }

  // ============ SSE 流式请求 ============
  async function sendMessage(text) {
    if (streaming) return;
    if (!text.trim()) return;

    streaming = true;
    DOM.send.disabled = true;
    DOM.input.value = "";
    autoResize();

    addUserMessage(text);
    const aiWrap = addAssistantMessage("智能客服");

    try {
      const resp = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Demo-User": userId || "guest",
        },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!resp.ok || !resp.body) {
        throw new Error("HTTP " + resp.status);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let bubble = aiWrap.querySelector(".bubble");
      let currentEvent = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          // SSE 规范：event: xxx 记录事件名，下一行 data: xxx 是数据
          if (line.startsWith("event:")) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith("data:") && currentEvent) {
            try {
              const payload = JSON.parse(line.slice(5).trim());
              handleEvent({ event: currentEvent, data: payload }, aiWrap, bubble);
            } catch (_) {
              // 忽略解析失败的行
            }
            // SSE 事件以空行分隔，处理完 data 后重置 event
            // （下一个 event: 行重新赋值即可）
          }
        }
      }
    } catch (err) {
      aiWrap.querySelector(".bubble").classList.remove("streaming");
      aiWrap.querySelector(".bubble").textContent =
        "抱歉，服务暂时不可用，请稍后重试。（" + err.message + "）";
    } finally {
      streaming = false;
      DOM.send.disabled = false;
      bubble = aiWrap.querySelector(".bubble");
      bubble && bubble.classList.remove("streaming");
      scrollToBottom();
    }
  }

  function handleEvent(ev, wrap, bubble) {
    switch (ev.event) {
      case "session":
        sessionId = ev.data.session_id;
        localStorage.setItem("lib_session_id", sessionId);
        break;
      case "route":
        // 路由事件：可用于显示通道（可选）
        break;
      case "token":
        bubble.textContent += ev.data.content;
        scrollToBottom();
        break;
      case "card":
        renderCard(wrap, ev.data.extra);
        if (ev.data.reply_type === "transfer") renderTransfer(wrap);
        break;
      case "slot_required":
      case "confirm":
        // 把 slot_fill / confirm 阶段的槽位要求信息附加到气泡后，方便用户查看
        // 主内容由 token 事件拼接
        break;
      case "sources":
        renderSources(wrap, ev.data.documents);
        break;
      case "done":
        bubble.classList.remove("streaming");
        // done 事件的 data 包含最终结果，若之前 token 没拼接（如一次性回复），兜底显示
        if (!bubble.textContent.trim() && ev.data && ev.data.answer) {
          bubble.textContent = ev.data.answer;
        }
        break;
      case "error":
      case "partial":
        bubble.classList.remove("streaming");
        bubble.textContent += "\n[错误] " + (ev.data.message || ev.data.reason || "未知错误");
        break;
    }
  }

  // ============ 输入框自适应高度 ============
  function autoResize() {
    DOM.input.style.height = "auto";
    DOM.input.style.height = Math.min(DOM.input.scrollHeight, 120) + "px";
  }

  // ============ 模式徽标 ============
  async function loadMode() {
    try {
      const resp = await fetch("/api/v1/admin/kb/info");
      const json = await resp.json();
      if (json.code === 0) {
        const d = json.data;
        const docCount = Array.isArray(d.documents) ? d.documents.length : 0;
        const chunkCount = typeof d.chunk_count === "number" ? d.chunk_count : d.index_size || 0;
        DOM.mode.textContent =
          `知识库 ${docCount} 篇 · ${chunkCount} 块`;
      } else {
        DOM.mode.textContent = "服务异常";
      }
    } catch (_) {
      DOM.mode.textContent = "无法连接服务";
    }
  }

  // ============ 登录 / 退出 ============
  function updateLoginUI() {
    if (userId) {
      DOM.login.textContent = `✓ ${userId}`;
      DOM.login.classList.add("logged-in");
      DOM.exit.classList.remove("hidden");
    } else {
      DOM.login.textContent = "👤 登录";
      DOM.login.classList.remove("logged-in");
      DOM.exit.classList.add("hidden");
    }
  }

  function logout() {
    if (streaming) return;
    userId = "";
    localStorage.removeItem("lib_user_id");
    updateLoginUI();
    const wrap = addAssistantMessage("智能客服");
    wrap.querySelector(".bubble").textContent = "已退出登录，当前为游客身份。";
  }

  function promptLogin() {
    const input = prompt(
      "体验登录（演示账号）：\n学号 2024001 / 2024002，密码 123456\n\n请输入学号：",
      userId || "2024001"
    );
    if (input === null) return;
    const id = input.trim();
    const pwd = prompt("请输入密码：", "123456");
    if (pwd === null) return;

    fetch("/api/v1/user/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: id, password: pwd }),
    })
      .then((r) => r.json())
      .then((json) => {
        if (json.code === 0) {
          userId = json.data.user_id;
          localStorage.setItem("lib_user_id", userId);
          updateLoginUI();
          const wrap = addAssistantMessage("智能客服");
          wrap.querySelector(".bubble").textContent = json.message;
        } else {
          alert(json.message);
        }
      })
      .catch(() => alert("登录服务异常"));
  }

  // ============ 新会话 ============
  function newChat() {
    if (streaming) return;
    sessionId = null;
    localStorage.removeItem("lib_session_id");
    DOM.list.innerHTML =
      `<div class="welcome-card">
        <h2>你好，我是图书馆智能客服 🤖</h2>
        <p>我可以帮你查书、查借阅、预约座位、解答图书馆规则。</p>
        <p>试试点击下方快捷问题，或直接输入你的问题。</p>
      </div>`;
  }

  // ============ 事件绑定 ============
  DOM.send.addEventListener("click", () => sendMessage(DOM.input.value));
  DOM.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(DOM.input.value);
    }
  });
  DOM.input.addEventListener("input", autoResize);
  DOM.newChat.addEventListener("click", newChat);
  DOM.login.addEventListener("click", promptLogin);
  DOM.actions.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) sendMessage(chip.dataset.msg);
  });
  DOM.exit.addEventListener("click", logout);

  // ============ 初始化 ============
  updateLoginUI();
  loadMode();
})();
