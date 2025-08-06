<template>
  <div class="home">
    <h1>Bienvenue sur l'application d'analyse IA</h1>
    <p>Utilisez le menu pour accéder au formulaire de prédiction des tickets Jira.</p>

    <h2>📥 Importer un fichier ZH12</h2>
    <form @submit.prevent="envoyerFichier">
      <input type="file" @change="onFileChange" accept=".xlsx,.csv" />
      <button type="submit">Envoyer</button>
    </form>

    <p v-if="message" :class="{ success: success, error: !success }">{{ message }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const fichier = ref(null)
const message = ref('')
const success = ref(false)

function onFileChange(e) {
  fichier.value = e.target.files[0]
}

async function envoyerFichier() {
  if (!fichier.value) {
    message.value = 'Aucun fichier sélectionné.'
    success.value = false
    return
  }

  const formData = new FormData()
  formData.append('excel_file', fichier.value)

  try {
    const response = await fetch('/api/api/import', {
      method: 'POST',
      body: formData
    })
    const data = await response.json()

    if (response.ok) {
      message.value = data.message + ` (${data.filename})`
      success.value = true
    } else {
      message.value = data.error || 'Erreur lors de l’envoi.'
      success.value = false
    }
  } catch (err) {
    message.value = 'Erreur réseau : ' + err.message
    success.value = false
  }
}
</script>

<style scoped>
.home {
  text-align: center;
  margin-top: 50px;
  font-family: 'Segoe UI', sans-serif;
}

form {
  margin-top: 30px;
}

input[type="file"] {
  margin-bottom: 10px;
}

button {
  padding: 8px 16px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

button:hover {
  background-color: #2980b9;
}

.success {
  color: green;
  margin-top: 10px;
}

.error {
  color: red;
  margin-top: 10px;
}
</style>
