<template>
  <div class="max">
    <v-container id="footer">
      <v-divider></v-divider>
      <v-row class="mt-1">
      <!-- Col 1 -->
        <v-col cols="10" md="10" sm="10">
        <p class="body-1 font-weight-thin">
          Universidad de los Andes <br>
          Mérida-Venezuela <br>
          &copy; 2021
        </p>
        </v-col>
        <v-col cols="2" md="2" sm="2" class="d-flex flex-row-reverse">
          <a href="http://www.ula.ve/" target="_blank">
            <v-img 
              alt="Logotipo" 
              src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952211/proyecto-grado/ula_wxoiku.png"
              lazy-src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952211/proyecto-grado/ula_wxoiku.png"
              max-height="90"
              max-width="56"
            >
            </v-img>
        </a>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>
<script>
  export default {
    name: 'Footer',
    data () {
      return {
        drawer: null,
      }
    },
  }
</script>
<style scoped>
  .max {
    max-width: 1200px;
    margin: auto;
  }
  .drawer {
    background-color: #F5F5F5!important;
    color: #2C2D48 !important;
    text-align: left;
    padding: 1.5rem;
  }
  .drawer h1, h3 {
    text-align: center !important;
    padding: 3% 2%;
  }
</style>