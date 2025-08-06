<template>
  <div class="dashboard">
    <h1>Tableau de bord</h1>

    <nav class="subnav">
      <router-link to="/dashboard/table">Table</router-link>
      <router-link to="/dashboard/Waterfall">Waterfall imputation</router-link>
      <router-link to="/dashboard/tablepredict">Table Prédictions</router-link>
      <router-link to="/dashboard/WaterfallPredict">Waterfall Prédictions</router-link>
    </nav>

    <!-- Affiche le dashboard principal ssi route = /dashboard -->
    <div
      v-if="isRootDashboard"
      id="superset-container"
      class="dashboard-frame"
    ></div>

    <router-view />
  </div>
</template>

<script setup>
import { useRoute } from 'vue-router'
import { computed, onMounted } from 'vue'
import { embedDashboard } from '@superset-ui/embedded-sdk'
import { SUPERSET_BASE_URL } from '@/config/superset.js'
import '../assets/DashboardView.css'

// UUID du dashboard récupéré dans Superset (embed section)
const dashboardId = 'af0bb59b-89d8-4825-9fc2-1b69c4b52d64'  // à adapter si différent

const isRootDashboard = computed(() => useRoute().path === '/dashboard')

const fetchGuestToken = async () => {
  const res = await fetch(`/guest_token/${dashboardId}`)
  const { token } = await res.json()
  return token
}

onMounted(async () => {
  if (!isRootDashboard.value) return

  await embedDashboard({
    id: dashboardId,
    supersetDomain: SUPERSET_BASE_URL,
    mountPoint: document.getElementById('superset-container'),
    fetchGuestToken,
    dashboardUiConfig: {
      hideTitle: false,
      filters: {
        expanded: true,
      }
    },
    iframeSandboxExtras: [
      'allow-top-navigation',
      'allow-popups-to-escape-sandbox'
    ],
    referrerPolicy: 'strict-origin-when-cross-origin'
  })
})
</script>
