<template>
  <Webcam />
</template>

<script lang="ts">
import Vue from 'vue'
import Webcam from '~/components/Webcam.vue'

export default Vue.extend({
  name: 'IndexPage',
  components: { Webcam },
})
</script>
