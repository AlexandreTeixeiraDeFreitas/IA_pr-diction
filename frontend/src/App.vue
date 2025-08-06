<template>
  <div id="app">
    <header class="navbar">
      <div class="container">
        <h1 class="logo">IA Tickets</h1>
        <nav class="nav-links">
          <router-link to="/" exact>Accueil</router-link>
          <router-link to="/dashboard">Dashboard</router-link>
          <router-link to="/tickets">Tickets</router-link>
        </nav>
      </div>
    </header>

    <router-view />
  </div>
</template>

<script setup>
import './assets/App.css'
</script>
