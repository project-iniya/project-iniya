<template>
  <div :class="styles.chatBoxContainer" @click="handleClick">
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
            <div :class="styles.messageText">
              {{ msg.content }}
            </div>

            <!-- attachments (optional) -->
            <div v-if="msg.images?.length" :class="styles.attachments">
              <img
                v-for="(img, i) in msg.images"
                :key="i"
                :src="img"
                :class="styles.imagePreview"
              />
            </div>

            <div :class="styles.timestamp">
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
  import { Send, Plus, ImagePlus, FilePlus, Loader, Info } from 'lucide-vue-next'
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

  function sendMSg() {
    if (message.value.trim() === "") return

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
          data: reader.result // base64 DataURL
        })
        if(res.status==='ok'){
          inputtedImages.value.push(res.path)
        }
      }

      reader.readAsDataURL(file)
    }

    // reset so same file can be picked again
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
        data: reader.result // base64 DataURL
      })
      if(res.status==='ok'){
        inputtedFiles.value.push(res.path)
      }
    }

    reader.readAsDataURL(file)

    // reset so same file can be picked again
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

  window.sendResponse = function (payload){
    payload.timestamp = new Date().toISOString()
    console.log("📨 Event from Python:", payload)
    messagesList.value.push(payload)
    console.log(messagesList.value)
    isWaitingForRply.value=false
    autoScroll()
  }
</script>