import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import TicketsView from '../views/TicketsView.vue'
import DashboardView from '../views/DashboardView.vue'
import TableView from '../views/dashboard/TableView.vue'
import TablePredictView from '../views/dashboard/TablePredictView.vue'
import Waterfall_prédictionsView from '../views/dashboard/Waterfall_prédictionsView.vue'
import WaterfallView from '../views/dashboard/WaterfallView.vue'


const routes = [
  { path: '/', name: 'Home', component: HomeView },
  { path: '/tickets', name: 'Tickets', component: TicketsView },
  {
    path: '/dashboard',
    component: DashboardView,
    children: [
      { path: 'table', component: TableView },
      { path: 'Waterfall', component: WaterfallView },
      { path: 'tablepredict', component: TablePredictView },
      { path: 'WaterfallPredict', component: Waterfall_prédictionsView },
    ]
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
