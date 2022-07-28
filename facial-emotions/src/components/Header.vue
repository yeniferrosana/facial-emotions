<template>
  <div class="max">
    <v-container id="nav">
      <v-row v-if="!$vuetify.breakpoint.smAndDown">
      <!-- Col 1 -->
        <v-col cols="12" md="2" sm="12" order="3" order-sm="1" class="d-flex flex-row">
        <router-link to="/">
          <v-img 
            alt="Logotipo" 
            src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952210/proyecto-grado/logo_ddn2dl.png"
            lazy-src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952210/proyecto-grado/logo_ddn2dl.png"
            max-height="50"
            max-width="50"
            class="ml-n1"
          >
          </v-img>
        </router-link>
        </v-col>
        <v-col 
          cols="12" 
          md="8" 
          sm="12"
          order="3"
          order-sm="1"
        >
          <v-spacer/>
        </v-col>
        <v-col cols="12" md="2" sm="12" order="3" order-sm="1" class="d-flex flex-row-reverse">
          <v-btn icon href="#" @click.stop="drawer = !drawer">
            <v-icon>mdi-information-outline</v-icon>
          </v-btn>
          <!--Configuración de info-->
          <v-navigation-drawer
            v-model="drawer"
            class="drawer"
            absolute
            temporary
            right
            width="30%"
          >
            <h2 class="text-center">Proyecto de Grado</h2>
            <h3>Aplicación del Aprendizaje Profundo para el Reconocimiento y Clasificación de Emociones Faciales</h3>
            <p class="text-justify">
              En este Proyecto de Grado, se implementa un Sistema de Reconocimiento de Emociones Faciales con algoritmos de Aprendizaje Profundo, que permite a los usuarios la clasificación automática de emociones en rostros de personas, mediante el vídeo en tiempo real captado por la cámara del dispositivo donde se ejecute la aplicación web. La tarea de clasificación de este sistema es realizado por un <a href="https://drive.google.com/drive/folders/1Ylv0mi7pRYCzx4HS2MZtzYa6_fwYQ1F8?usp=sharing"> modelo entrenado </a> con Aprendizaje Profundo, implementado con dos algoritmos distintos, por lo que se opta por hacer una validación con un caso de estudio, enfocado en las emociones de influencers en vídeos de reacciones.
            </p>
            
          </v-navigation-drawer>
                    <v-btn 
            icon
            href="https://github.com/yeniferramirez11/proyecto-grado/tree/develop" 
            target="_blank"
          >
            <v-icon>mdi-github</v-icon>
          </v-btn>
     
          <v-btn 
            icon
            href="https://github.com/yeniferramirez11/proyecto-grado/blob/master/redaccion/tesis/documento.pdf" 
            target="_blank"
          >
            <v-icon>mdi-file-document</v-icon>
          </v-btn>
        </v-col>
      </v-row>


      <!--Menu Para Tablets y Movil-->
      <v-row v-if="$vuetify.breakpoint.smAndDown">
      <!-- Col 1 -->
        <v-col sm="6" class="d-flex flex-row">
        <router-link to="/">
          <v-img 
            alt="Logotipo" 
            src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952210/proyecto-grado/logo_ddn2dl.png"
            lazy-src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952210/proyecto-grado/logo_ddn2dl.png"
            max-height="50"
            max-width="50"
          >
          </v-img>
        </router-link>
        </v-col>
        <v-col sm="6" class="d-flex flex-row-reverse">
          <v-btn icon href="#" @click.stop="drawer = !drawer">
            <v-icon>mdi-information-outline</v-icon>
          </v-btn>
          <!--Configuración de info-->
          <v-navigation-drawer
            v-model="drawer"
            class="drawer"
            absolute
            temporary
            left
            width="70%"
          >
            <h2 class="text-center">Proyecto de Grado</h2>
            <h3>Aplicación del Aprendizaje Profundo para el Reconocimiento y Clasificación de Emociones Faciales</h3>
            <p>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut efficitur sapien nec nisi vulputate mollis. Quisque hendrerit suscipit urna nec dignissim. Suspendisse tincidunt sed erat nec aliquet. Sed blandit et diam in sagittis. Proin quis mi pretium nunc posuere viverra. Integer et magna lorem. Morbi ut nulla fermentum, ullamcorper felis non, tempor elit. Proin sodales lorem a augue feugiat sodales. Vivamus volutpat in odio condimentum convallis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
            </p>
            
          </v-navigation-drawer>
          <v-btn 
            icon
            href="https://github.com/yeniferramirez11/proyecto-grado/tree/develop" 
            target="_blank"
          >
            <v-icon>mdi-github</v-icon>
          </v-btn>
     
          <v-btn 
            icon
            href="https://github.com/yeniferramirez11/proyecto-grado/blob/master/redaccion/tesis/documento.pdf" 
            target="_blank"
          >
            <v-icon>mdi-file-document</v-icon>
          </v-btn>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>
<script>
  export default {
    name: 'Menu',
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