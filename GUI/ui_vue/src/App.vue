<template>
  <div :class="styles.mainAPP">
    <MainSec />
  </div>
</template>

<script setup>
  import MainSec from './components/main_section/mainSec.vue';
  import styles from './App.module.css';
</script>