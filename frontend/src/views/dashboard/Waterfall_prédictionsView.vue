<template>
  <div>
    <div id="superset-container" class="dashboard-frame"></div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { embedDashboard } from '@superset-ui/embedded-sdk'
import { SUPERSET_BASE_URL } from '@/config/superset.js'
import '@/assets/TableView.css'

// UUID du dashboard (embed ID de Superset)
const dashboardId = '649c521b-ea5d-48d0-bc89-14981f258b32'

const fetchGuestToken = async () => {
  const res = await fetch(`/guest_token/${dashboardId}`)
  const { token } = await res.json()
  return token
}

onMounted(() => {
  window.addEventListener('message', event => {
    if (
      event.origin === SUPERSET_BASE_URL &&
      event.data?.type === 'embed-superset-height'
    ) {
      const iframe = document.querySelector('iframe')
      if (iframe) {
        iframe.style.height = `${event.data.height}px`
      }
    }
  })

  embedDashboard({
    id: dashboardId,
    supersetDomain: SUPERSET_BASE_URL,
    mountPoint: document.getElementById('superset-container'),
    fetchGuestToken,
    iframeSandboxExtras: ['allow-same-origin', 'allow-scripts'],
    dashboardUiConfig: {
      hideTitle: false,
      filters: { expanded: true },
    },
  })
})
</script>