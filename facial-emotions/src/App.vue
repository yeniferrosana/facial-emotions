<template>
  <v-app>
    <VueElementLoading
        v-if="$route.name === 'Emotion'"
        :active="fullscreen"
        is-full-screen
        color="#F5F5F5"
        background-color="#30475E"
        spinner="bar-fade-scale"
        size="64"
        duration="1.2"
    />
    <VueElementLoading
        v-if="$route.name === 'Action'"
        :active="fullscreen"
        is-full-screen
        color="#F5F5F5"
        background-color="#F05454"
        spinner="bar-fade-scale"
        size="64"
        duration="1.2"
    />
    <v-main>  
      <!-- Componente 'Menu' -->
      <Menu/>  
 
        <router-view> 
 
          <v-content>
 
           
 
          </v-content>      
 
        </router-view>     
 
      <!-- Componente 'Footer' -->
      <Footer/>
 
    </v-main>
  </v-app>
</template>

<script type="text/javascript">
import Menu from './components/Header.vue'
import Footer from './components/Footer.vue'
import VueElementLoading from "vue-element-loading"

export default {
  data () {
    return {
      fullscreen: false,
    }
  },
  components: {
    Menu,
    Footer,
    VueElementLoading
  },
  mounted () {
   this.fullscreen = true
   setTimeout(() => { this.fullscreen = false }, 7000)
  }
}
</script>

<style lang="scss">
@import url('https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Lato&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Poppins:wght@300&display=swap');


#app {
  font-family: 'Poppins', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2C2D48;
  background-color: #F5F5F5;
}
#nav{
  padding: 20px 20px;
  background-color: #F5F5F5!important;
}

#footer {
  margin: auto;
  background-color: #F5F5F5!important;
}

a {
  font-weight: bold;
  color: #737373 !important;
  text-transform: none;
  text-decoration: none;
}

h1 {
  font-family: 'Anton', sans-serif;
  font-size: 48px;
  font-weight: normal;
}

h2 {
  font-family: 'Anton', sans-serif;
  font-size: 36px;
  font-weight: normal;
}

</style>
