<template>
  <div class="pt-4">
		<v-container class="px-4 max">
			<v-row>
				<!-- Col 1 -->
				<v-col cols="12" md="7" sm="12" order="3" order-sm="1">
					<h1>Reconocimiento y Clasificación de Emociones Faciales</h1>
					<!--<p class="text-left py-4">
            La capacidad que tiene el ser humano para detectar e identificar emociones es muy buena, y hace posible que se puedan entender los unos a los otros <a href="https://upcommons.upc.edu/handle/2117/129220" target="_blank">[1]</a>. Aunque, a veces esta tarea se torna muy difícil, debido al movimiento no rígido de los músculos faciales <a href="https://d1wqtxts1xzle7.cloudfront.net/48685809/COISINT2016_Emociones_E-075-with-cover-page-v2.pdf?Expires=1632519372&Signature=WaFvheCROHQj7ErAd0D-SdZ589bUN0wn6mluBl7I2xKDVpyvqPifALM3EiBR~dv6zmLq2dz-VauH6VKl74EKsBpHlyb0qgKMvADydmhXWWn-Kg8MFNdg0Uoun72FTIvcxNkopmSXownE1VPPxlYzJn2Sj4sJ3U9xfnvjhfSxJaduyCcqhoyhHn7TF5lQX8gSpXROFd3vcJPfG9SaTGB9~prlbc2l~ET1ufPIlzQwVpLPnx9S3eKu85XcI-aZrqa48t3R1HiYCxbBnXPCcCyB3p~zvKgYQA0jvY~IQ4--s-w6liAYLfgKXB30SgSLqB7jVHjMYBWkMWfSxjxQAGOloQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA">[2]</a> que hace que no se pueda reconocer una emoción espontánea, de una artificial.
					</p>-->
          <p class="mt-2 text-justify">
            Actualmente, y tras la popularización de la Inteligencia Artificial, muchos investigadores del área se han enfocado en resolver los problemas relacionados con el reconocimiento facial de emociones en personas, ya sea para ayudar a otras personas a identificar las emociones reales, o a la Computación Afectiva para mejorar la relación hombre-computador.
          </p>
          <p class="text-justify">
            Investigaciones realizadas han afirmado que la Inteligencia Artificial y en específico el Aprendizaje Profundo, son herramientas que permiten mejorar los tiempos de reconocimientos faciales automáticos, un alto porcentaje de esta investigaciones presentan excelentes resultados en entornos controlados, pero no en espacios no controlados, dónde es más común que cualquier dispositivo inteligente o ser humano interactúe.
          </p>
				</v-col>
				<v-col cols="12" md="5" sm="12" order="3" order-sm="1">
				<v-row
          v-for="(item, i) in items"
          :key="i"
          class="mt-lg-1"
        >
				<v-card
            :color="item.color"
            dark
            width="100%"
            class="pa-2"
          >
            <div class="d-flex flex-no-wrap justify-space-between">
              <v-avatar
                v-if="!$vuetify.breakpoint.smAndDown"
                class="ma-2"
                size="125"
                tile
              >
                <v-img :src="item.src" :lazy-src="item.src"></v-img>
              </v-avatar>
              
              <v-row
                align="center"
              >

                <v-avatar
                  v-if="$vuetify.breakpoint.smAndDown"
                  class="mx-4 mt-3 align-center justify-center"
                  size="100"
                  tile
                >
                  <v-img :src="item.src" :lazy-src="item.src"></v-img>
                </v-avatar>
              
                <v-card-title
                  class="text-h5 text"
                  v-text="item.title"
                ></v-card-title>

                <v-card-subtitle v-text="item.artist"></v-card-subtitle>

                <v-card-actions class="grow">
                  <v-spacer></v-spacer>
                  <router-link :to="item.route">
                  <v-btn
                    class="mt-n10"
                    fab
                    icon
                  >
                    <v-icon small>mdi-arrow-right</v-icon>
                  </v-btn>
                </router-link>
                </v-card-actions>
              </v-row>
              
            </div>
          </v-card>
          </v-row>
				</v-col>
			</v-row>
    </v-container>
  </div>
</template>

<script>

export default {
  name: 'Home',
  data: () => ({
    items: [
      {
        color: '#F05454',
        src: 'https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952260/proyecto-grado/au_tdwnij.gif',
        title: 'Seguimiento Facial',
        artist: 'Es un método nativo de los humanos, con el cual es posible identificar el rostro de una persona por medio de una imagen o el cuadro de un video.',
        route: '/action',
      },
      {
        color: '#30475E',
        src: 'https://res.cloudinary.com/yeniferramirez11/image/upload/v1632952250/proyecto-grado/ref_ut0fvy.gif',
        title: 'Reconocimiento de Emociones Faciales',
        artist: 'Es la capacidad de reconocer formas básicas de expresión afectiva, la cual se muestra en los rostros de las personas.',
        route: '/emotion',
      },
    ],
  }),
}
</script>

<style scoped>
  .max {
    max-width: 1200px;
    margin: auto;
  }
  .v-card:hover{
		opacity: 0.6;
  }
  .v-card {
		transition: opacity .4s ease-in-out;
		margin: 1rem 1rem;
		border-radius: 1em !important;
	}
</style>
