import { createApp } from 'vue'
import App from './App.vue'
import router from './router'  // ⬅️ C’est ici que router/index.js est importé

const app = createApp(App)

app.use(router)  // ⬅️ Le routeur est intégré à l'application Vue

app.mount('#app')
