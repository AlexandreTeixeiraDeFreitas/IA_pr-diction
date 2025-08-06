<template>
  <div>
    <div v-for="(ligne, index) in lignes" :key="index" class="ligne">
      <input v-model="ligne.matricule" placeholder="Matricule" type="text" />
      <input v-model="ligne.date_commencement" type="date" />
      <input v-model="ligne.tickets" placeholder="Tickets Jira" type="text" />
      <button @click="supprimerLigne(index)">Supprimer</button>
    </div>

    <div class="actions">
    <button @click="ajouterLigne">Ajouter une ligne</button>
    <button @click="envoyer">Analyser</button>
    </div>

    <div v-if="loading"><em>Traitement en cours...</em></div>

    <div v-if="resultats.length">
      <h3>Prédictions :</h3>
      <table border="1">
        <thead>
          <tr>
            <th>Clé</th>
            <th>Résumé</th>
            <th>Prédiction</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in resultats" :key="i">
            <td>{{ r.key || '-' }}</td>
            <td>{{ r['fields.summary'] || '-' }}</td>
            <td>{{ r.predict }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="messageErreur" style="color:red;">{{ messageErreur }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import '../assets/TicketForm.css'

const lignes = ref([
  { matricule: '', date_commencement: '', tickets: '' }
])

const resultats = ref([])
const loading = ref(false)
const messageErreur = ref('')

function ajouterLigne() {
  lignes.value.push({ matricule: '', date_commencement: '', tickets: '' })
}

function supprimerLigne(index) {
  lignes.value.splice(index, 1)
}

async function envoyer() {
  resultats.value = []
  messageErreur.value = ''
  loading.value = true

  const appels = lignes.value.map(async (ligne) => {
    const tickets = ligne.tickets.trim().split(/[,\s]+/)
    if (!tickets.length || !ligne.matricule || !ligne.date_commencement) return null

    const payload = {
      tickets,
      matricule: ligne.matricule,
      date_commencement: ligne.date_commencement
    }

    try {
      const res = await fetch('/api/api/tickets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()
      return res.ok ? data : [{ key: 'Erreur', 'fields.summary': '', predict: data.error }]
    } catch (e) {
      return [{ key: 'Exception', 'fields.summary': '', predict: e.message }]
    }
  })

  const réponses = await Promise.all(appels)
  réponses.filter(Boolean).forEach(r => resultats.value.push(...r))

  if (!resultats.value.length) {
    messageErreur.value = 'Aucune donnée traitée.'
  }

  loading.value = false
}
</script>
