<template>
  <div :class="styles.mainContainer">
    <div :class="[styles.leftPane, isLeftPaneCollapsed && styles.collapsed]">
      <LeftPane @pane="handlePane" @optionSelected="handleOptionSelected" :collapsed="isLeftPaneCollapsed" />
    </div>
    <div :class="styles.mainPane">
      <chatBox v-if="mainPanelContent === 'newChat'" />
    </div>
  </div>
</template>

<script setup>
  import styles from './mainSec.module.css';
  import LeftPane from '../left_pane/leftPane.vue';
  import chatBox from './chat_box/chatBox.vue';
  import { ref } from 'vue';

  const isLeftPaneCollapsed = ref(false);
  const mainPanelContent = ref('newChat');

  function handlePane(msg) {
    if (msg === 'close') 
      isLeftPaneCollapsed.value = true;
    else if (msg === 'open') 
      isLeftPaneCollapsed.value = false;
    
  }
  function handleOptionSelected(option) {
    mainPanelContent.value = option;
    console.log('Option selected:', option);
  }
</script>