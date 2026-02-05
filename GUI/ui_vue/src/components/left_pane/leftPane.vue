<template>
  <div
    :class="[styles.leftPane, collapsed ? styles.leftPaneCollapsed : styles.leftPaneExpanded]"
  >
    <!-- Title Bar -->
    <div :class="styles.paneTitleBar">
      <div
        :class="styles.logoWrapper"
        @click="collapsed && handlePane('open')"
      >
        <!-- Default Logo -->
        <img
          src="/src/assets/logo.png"
          :class="[styles.logoDefault, collapsed ? styles.logoCollapsed : styles.logoPaneOpen]"
          @click="!collapsed && handleOptionSelected('newChat')"
          alt="Logo"
        />

      </div>
      <button
        :class="collapsed ? styles.iconCollapsed : styles.icon"
        @click="handlePane(collapsed ? 'open' : 'close')"
      >
        <PanelLeftOpen v-if="collapsed" :size="22" />
        <PanelLeftClose v-else :size="22" />
      </button>
    </div>

    <!-- Options -->
    <div :class="styles.optionBlocks">
      <div
        v-for="item in options"
        :key="item.id"
        :class="styles.optionBlock"
      >
        <button
          :class="styles.Btn"
          @click="handleOptionSelected(item.action)"
        >
          <span :class="styles.icon">
            <component :is="item.icon" :size="16" />
          </span>
          <span :class="styles.optionText">
            {{ item.label }}
          </span>
        </button>
      </div>
    </div>

    <div :class="styles.chatHistoryBlock" v-if="!collapsed">
      <div :class="styles.textChat">
        <div :class="styles.chatHistoryTitle">
          <span :class="styles.HeaddingText">Text Chats</span>
          <button :class="styles.Chevron"  @click="handleRotate('text', isTextChatHistoryOpen ? 'close' : 'open')" ><ChevronUp :size="20" :class="{ [styles.rotated]: isTextChatHistoryOpen }"/></button>
        </div>
      </div>
      <div :class="styles.audioChat">
        <div :class="styles.chatHistoryTitle">
          <span :class="styles.HeaddingText">Audio Chats</span>
          <button :class="styles.Chevron"  @click="handleRotate('audio', isAudioChatHistoryOpen ? 'close' : 'open')" ><ChevronUp :size="20" :class="{ [styles.rotated]: isAudioChatHistoryOpen }"/></button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import styles from './leftPane.module.css'
import { ref } from 'vue'
import {
  PanelLeftClose,
  PanelLeftOpen,
  MessageSquarePlus,
  AudioLines,
  ChevronUp,
  File,
  Image,
  ImageUp,
  Settings
} from 'lucide-vue-next'

defineProps({
  collapsed: Boolean
})


const isTextChatHistoryOpen = ref(false)
const isAudioChatHistoryOpen = ref(false)
const emit = defineEmits(['pane', 'optionSelected'])

const options = [
  {
    id: 1,
    label: 'New Chat',
    action: 'newChat',
    icon: MessageSquarePlus
  },
  {
    id: 2,
    label: 'New Voice Chat',
    action: 'newVoiceChat',
    icon: AudioLines
  },
  {
    id: 5,
    label: 'Generated Images',
    action: 'generatedImages',
    icon: Image
  },
  {
    id: 6,
    label: 'Settings',
    action: 'settings',
    icon: Settings
  }
]

function handleRotate(type, action) {
  if (type === 'text') {
    if (action === 'open') {
      isTextChatHistoryOpen.value = true
      isAudioChatHistoryOpen.value = false
    } else {
      isTextChatHistoryOpen.value = false
      isAudioChatHistoryOpen.value = false
    }
  } else if (type === 'audio') {
    if (action === 'open') {
      isAudioChatHistoryOpen.value = true
      isTextChatHistoryOpen.value = false
    } else {
      isAudioChatHistoryOpen.value = false
      isTextChatHistoryOpen.value = false
    }
  }

}

function handlePane(state) {
  emit('pane', state)
}

function handleOptionSelected(option) {
  emit('optionSelected', option)
}
</script>
