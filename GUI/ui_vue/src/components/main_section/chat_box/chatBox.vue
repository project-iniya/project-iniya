<template>
  <div :class="styles.chatBoxContainer">
    <div :class="styles.chatHeader">
      <div :class="styles.headerRow">
        <div :class="styles.headerSide">
          <div :class="styles.headerElement">
            <Info :size="18"/>
            <span>
              Info
            </span>
          </div>
        </div>
        <div :class="styles.headerSide">
          <div :class="styles.headerElement">
            <FilePlus :size="18"/>
            <span>
              Files
            </span>
          </div>
          <div :class="styles.headerElement">
            <ImagePlus :size="18"/>
            <span>
              Images
            </span>
          </div>
        </div>
      </div>
    </div>
    <div :class="styles.chatContent" ref="chatContent">
      <div :class="styles.chatInner" v-if="!newChat">
        <div
          v-for="(msg, index) in messagesList"
          :key="index"
          :class="[
            styles.messageRow,
            msg.source === 'user' ? styles.userRow : styles.botRow
          ]"
        >
          <div
            :class="[
              styles.messageBubble,
              msg.source === 'user' ? styles.userBubble : styles.botBubble
            ]"
          >
            <!-- ── BOT MESSAGE ── -->
            <template v-if="msg.source === 'bot'">

              <!-- Step list -->
              <div v-if="msg.steps?.length" :class="styles.stepList">
                <div
                  v-for="step in msg.steps"
                  :key="step.id"
                  :class="[
                    styles.stepItem,
                    step.failed ? styles.stepFailed :
                    step.done   ? styles.stepDone   : styles.stepRunning,
                    (step.input || step.result || step.fullDetail) ? styles.stepClickable : ''
                  ]"
                  @click="step.expanded = !step.expanded"
                >
                  <!-- Header row -->
                  <div :class="styles.stepHeader">
                    <Loader     v-if="!step.done && !step.failed" :size="14" :class="styles.spinning" />
                    <XCircle    v-else-if="step.failed"           :size="14" :class="styles.stepError" />
                    <CheckCheck v-else                            :size="14" :class="styles.stepCheck" />
                    <span :class="styles.stepLabel">{{ step.label }}</span>
                    <span v-if="step.attempt > 1" :class="styles.stepAttempt">#{{ step.attempt }}</span>
                    <span v-if="step.detail && !step.expanded" :class="styles.stepDetail">{{ step.detail }}</span>
                    <ChevronDown
                      v-if="step.input || step.result || step.fullDetail"
                      :size="12"
                      :class="[styles.stepChevron, step.expanded ? styles.chevronOpen : '']"
                    />
                  </div>

                  <!-- Expanded panel — sits below header -->
                  <div v-if="step.expanded" :class="styles.stepExpanded">
                    <div v-if="step.fullDetail">
                      <span :class="styles.stepExpandLabel">Details</span>
                      <p :class="styles.stepFullDetail">{{ step.fullDetail }}</p>
                    </div>
                    <div v-if="step.input">
                      <span :class="styles.stepExpandLabel">Input</span>
                      <pre :class="styles.stepCode">{{ step.input }}</pre>
                    </div>
                    <div v-if="step.result">
                      <span :class="styles.stepExpandLabel">{{ step.failed ? 'Error' : 'Output' }}</span>
                      <pre :class="styles.stepCode">{{ step.result }}</pre>
                    </div>
                    <div v-if="step.reason">
                      <span :class="styles.stepExpandLabel">Failure reason</span>
                      <pre :class="styles.stepCode">{{ step.reason }}</pre>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Divider between steps and final answer -->
              <hr v-if="msg.steps?.length && msg.content" :class="styles.stepDivider" />

              <!-- Typing indicator while pending and no content yet -->
              <div v-if="msg.pending && !msg.content" :class="styles.typingDots">
                <span /><span /><span />
              </div>

              <!-- Final response -->
              <div
                v-if="msg.content"
                :class="styles.messageText"
                v-html="renderMarkdown(msg.content)"
              />

            </template>

            <!-- ── USER MESSAGE ── -->
            <template v-else>
              <div
                :class="styles.messageText"
                v-html="renderMarkdown(msg.content)"
              />
            </template>

            <!-- Attachments -->
            <div v-if="msg.images?.length" :class="styles.attachments">
              <img
                v-for="(img, i) in msg.images"
                :key="i"
                :src="img"
                :class="styles.imagePreview"
              />
            </div>

            <div :class="styles.timestamp" v-if="msg.source === 'user' || (msg.source === 'bot' && !msg.pending)">
              {{ new Date(msg.timestamp).toLocaleTimeString() }}
            </div>
          </div>
        </div>
      </div>
      <div v-else :class="styles.newChatTextWrapper">
        <p :class="styles.newChatText">
          {{ greeting }}
        </p>
      </div>
    </div>
    <div :class="styles.chatInputSection">
      <div :class="styles.inputWrapper">
        <div :class="styles.popupWrapper" ref="popupWrapper">
          <input ref="imageInput" type="file" accept="image/*" multiple hidden @change="onImageSelect"/>
          <input ref="fileInput" type="file" multiple hidden @change="onFileSelect"/>
          <div :class="[styles.attachmentPopup ,isattachemntPopOpen ? styles.attachmentPopupOpen : styles.attachmentPopupClosed]" >
            <ul :class="styles.attachmentList">
              <li :class="styles.attachmentItem" @click="openImages"><ImagePlus :size="16" /> Images </li>
              <li :class="styles.attachmentItem" @click="openFiles"><FilePlus :size="16" /> Files</li>
            </ul>
          </div>
          <div :class="styles.chatInputActions ">
            <button :class="[styles.Button, styles.leftAction, !isattachemntPopOpen ?styles.attachmentBtnClosed:'']" :style="{ height: `${height}px` }" @click.stop="togglePopup">
              <Plus :size="20" />
            </button>
          </div>
        </div>
        <textarea ref="textarea" v-model="message" :rows="rows" :class="styles.chatInput" placeholder="Type your message..." @input="autoGrow" />
        <div :class="styles.chatInputActions">
          <button :class="[styles.Button, styles.rightAction]" :style="{ height: `${height}px` }" @click="sendMSg()">
            <Send v-if="!isWaitingForRply" :size="20" />
            <Loader v-else :size="20" :class="styles.spinning" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
  import { ref, nextTick, onMounted, onBeforeUnmount } from 'vue'
  import styles from './chatBox.module.css'
  import { Send, Plus, ImagePlus, FilePlus, Loader, Info, CheckCheck, XCircle, ChevronDown } from 'lucide-vue-next'
  import MarkdownIt from "markdown-it"
  import hljs from "highlight.js"
  import "highlight.js/styles/github-dark.css"

  const newChat = ref(true)
  const textarea = ref(null)
  const message = ref("")
  const rows = ref(1)
  const height = ref(40)
  const isattachemntPopOpen = ref(false)
  const popupWrapper = ref(null)
  const imageInput = ref(null)
  const fileInput = ref(null)
  const messagesList = ref([])
  const inputtedFiles = ref([])
  const inputtedImages = ref([]) 
  const isWaitingForRply = ref(false)
  const chatContent = ref(null)
  const greeting = ref("")

  function autoGrow() {
    nextTick(() => {
      const el = textarea.value
      if (!el) return

      if (message.value === "") {
        rows.value = 1
        height.value = 40
        return
      }

      const style = window.getComputedStyle(el)
      const lineHeight = parseFloat(style.lineHeight)

      const padding =
        parseFloat(style.paddingTop) +
        parseFloat(style.paddingBottom)

      const visualLines = Math.ceil(
        (el.scrollHeight - padding) / lineHeight
      )

      rows.value = Math.min(
        8,
        Math.max(1, visualLines)
      )
      height.value = Math.min(
        180,
        Math.max(40, el.scrollHeight)
      )
    })
  }

  function autoScroll(){
    nextTick(() => {
      const el = chatContent.value
      if (el) el.scrollTop = el.scrollHeight
    })
  }

  const md = new MarkdownIt({
    html: false,
    linkify: true,
    typographer: true,
    breaks: true,
    highlight(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return `<pre class="hljs"><code>${
            hljs.highlight(code, { language: lang }).value
          }</code></pre>`
        } catch (_) {}
      }
      return `<pre class="hljs"><code>${md.utils.escapeHtml(code)}</code></pre>`
    }
  })

  function renderMarkdown(text) {
    if (!text) return ""
    return md.render(text)
  }

  async function sendMSg() {
    if (message.value.trim() === "") return

    if (newChat.value)
      await window.pywebview.api.change_chat({new:true, chatID:null})

    if (newChat.value) newChat.value = false

    let msg = {
      type:'message',
      source:'user', 
      timestamp: new Date().toISOString(), 
      content:message.value.trim(),
      files:inputtedFiles.value,
      images: inputtedImages.value
    }
    messagesList.value.push(msg)
    if (window.pywebview){
      window.pywebview.api.sendQuestion(msg)
      isWaitingForRply.value=true
      autoScroll()
    }
    message.value = ""
    inputtedFiles.value = []
    inputtedImages.value = []
    autoGrow()
  }

  function togglePopup() {
    isattachemntPopOpen.value = !isattachemntPopOpen.value
  }

  function handleClickOutside(e) {
    if (!popupWrapper.value) return
    if (!popupWrapper.value.contains(e.target)) {
      isattachemntPopOpen.value = false
    }
  }

  function openImages() {
    isattachemntPopOpen.value = false
    imageInput.value?.click()
  }

  function openFiles() {
    isattachemntPopOpen.value = false
    fileInput.value?.click()
  }

  async function onImageSelect(e) {
    const files = e.target.files
    if (!files || files.length === 0) return

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const reader = new FileReader()

      reader.onload = async () => {
        let res = await window.pywebview.api.receive_image({
          name: file.name,
          type: file.type,
          data: reader.result
        })
        if(res.status==='ok'){
          inputtedImages.value.push(res.path)
        }
      }

      reader.readAsDataURL(file)
    }

    e.target.value = ""
  }

  function onFileSelect(e) {
    const file = e.target.files[0]
    if (!file) return

    const reader = new FileReader()

    reader.onload = async () => {
      let res = await window.pywebview.api.receive_file({
        name: file.name,
        type: file.type,
        data: reader.result
      })
      if(res.status==='ok'){
        inputtedFiles.value.push(res.path)
      }
    }

    reader.readAsDataURL(file)
    e.target.value = ""
  }

  function getGreeting({ name = "", date = new Date() } = {}) {
    const hour = date.getHours();

    let timeLabel = "";
    if (hour < 12) timeLabel = "Good morning";
    else if (hour < 18) timeLabel = "Good afternoon";
    else timeLabel = "Good evening";

    const neutral = [
      "How can I help you today?",
      "What can I do for you?",
      "How may I assist you?",
      "What would you like to talk about?",
      "Need help with something?"
    ];

    const templates = [
      `${timeLabel}${name ? `, ${name}` : ""}!`,
      `${timeLabel}${name ? `, ${name}` : ""}. How can I help?`,
      `${name ? `${name}, ` : ""}how can I help you today?`,
      `How can I help you today${name ? `, ${name}` : ""}?`,
      neutral[Math.floor(Math.random() * neutral.length)]
    ];

    return templates[Math.floor(Math.random() * templates.length)];
  }

  onMounted(() => {
    if (newChat.value) greeting.value = getGreeting({name:"asd"})

    document.addEventListener('mousedown', handleClickOutside)

    if (window.pywebview) {
      initPywebview()
    } else {
      window.addEventListener('pywebviewready', initPywebview)
    }
  })

  function initPywebview() {
    console.log("✅ pywebview ready")

    window.pywebview.api.test_method("Hello from Vue!")
      .then(res => {
        console.log("Python replied:", res)
      })
  }

  onBeforeUnmount(() => {
    document.removeEventListener('mousedown', handleClickOutside)
  })

  // ── Python-callable window functions ──────────────────────────────────────

  // Called once when bot starts responding — creates the pending bot message
  window.startBotMessage = function () {
    console.log("🤖 [startBotMessage] called")
    const botMsg = {
      source: 'bot',
      timestamp: new Date().toISOString(),
      steps: [],
      content: null,
      pending: true
    }
    messagesList.value.push(botMsg)
    console.log("🤖 [startBotMessage] messagesList length:", messagesList.value.length)
    autoScroll()
  }

  // Called for each tool/thinking step
  window.addStep = function ({ label, detail = null, fullDetail = null, attempt = 1, input = null }) {
    const msg = messagesList.value.at(-1)
    if (!msg || msg.source !== 'bot') return
    msg.steps.push({ 
      label, detail, fullDetail: fullDetail || detail, attempt,
      input: input ? JSON.stringify(input, null, 2) : null,
      result: null,
      done: false, failed: false,
      expanded: false,
      id: Date.now() 
    })
    autoScroll()
  }

  // Called to mark a step as complete
  window.completeStep = function ({ label, result = null }) {
    const msg = messagesList.value.at(-1)
    if (!msg) return
    const step = [...msg.steps].reverse().find(s => s.label === label && !s.done)
    if (step) {
      step.done = true
      step.result = result
    }
  }

  window.failStep = function ({ label, reason = null, result = null }) {
    const msg = messagesList.value.at(-1)
    if (!msg) return
    // find last undone step with this label — don't push duplicate
    const step = [...msg.steps].reverse().find(s => s.label === label && !s.done)
    if (step) {
      step.done = true
      step.failed = true
      step.reason = reason
      step.result = result
    }
    // if already marked failed by toolDone, just update reason if missing
    else {
      const existing = [...msg.steps].reverse().find(s => s.label === label && s.failed)
      if (existing && !existing.reason) existing.reason = reason
    }
    autoScroll()
  }

  // Called at the end with the final response text
  window.finalizeResponse = function ({ content }) {
    console.log("📩 [finalizeResponse] content:", content)
    const msg = messagesList.value.at(-1)
    if (!msg) {
      console.warn("⚠️ [finalizeResponse] no last message found")
      return
    }
    console.log("📩 [finalizeResponse] target msg:", msg)
    msg.content = content
    msg.pending = false
    msg.timestamp = new Date().toISOString()
    isWaitingForRply.value = false
    autoScroll()
  }

  // Legacy fallback — simple response with no steps
  window.sendResponse = function (payload) {
    console.log("📨 [sendResponse] payload:", payload)
    payload.timestamp = new Date().toISOString()
    payload.steps = payload.steps ?? []
    payload.pending = false
    messagesList.value.push(payload)
    isWaitingForRply.value = false
    autoScroll()
  }
</script>