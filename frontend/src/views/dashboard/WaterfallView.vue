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

// UUID fourni dans Superset Embed (pas un slug)
const dashboardId = 'b09211b4-d129-40e2-9e01-ef4fec903351'

const fetchGuestToken = async () => {
  const res = await fetch(`/guest_token/${dashboardId}`)
  const { token } = await res.json()
  return token
}

// Ajuste dynamiquement la hauteur en écoutant les messages du SDK Superset
onMounted(() => {
  window.addEventListener('message', event => {
    if (
      event.origin === SUPERSET_BASE_URL &&
      event.data?.type === 'embed-superset-height'
    ) {
      const iframe = document.querySelector('iframe')
      if (iframe) {
        iframe.style.height = event.data.height + 'px'
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