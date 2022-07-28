<template>
  <div class="pt-4 max">
		<v-container class="px-4">
			<v-row>
				<!-- Col 1 -->
				<v-col 
					v-if="$vuetify.breakpoint.smAndDown"
					cols="12"
					sm="12"
				>
					<h1 class="text-center">Reconocimiento de Emociones Faciales</h1>
				</v-col>
				<v-col cols="12" md="6" sm="12">
					<Emotion/>
				</v-col>
				<v-col 
					v-if="$vuetify.breakpoint.smAndDown"
					cols="12"
					sm="12"
				>
					<p>
						Este componente realiza una extracción de los AU’s para luego procesar estos puntos y clasificar según las emociones básicas de Paul Ekman y la expresiń neutral, esta clasificación se lleva a cabo a través de uno de los modelos entrenados en esta investigación. 
						<br>
						Este modelo recibe el array de puntos extraídos por el face-landmarks-detection y predice cual es el emoción resultante tomada de la captura de vídeo del sistema, además de resaltar el contorno del rostro del evaluado.
					</p>
				</v-col>
				<!-- Web Normal -->
				<v-col 
					v-if="!$vuetify.breakpoint.smAndDown"
					cols="12"
					md="6"
					sm="12"
					order="3"
					order-sm="1"
				>
					<h1>Reconocimiento de Emociones Faciales</h1>
					<p>
						<p class="text-justify mt-4 mr-4">
						Este componente realiza una extracción de los AU’s para luego procesar estos puntos y clasificar según las emociones básicas de Paul Ekman y la expresiń neutral, esta clasificación se lleva a cabo a través de uno de los modelos entrenados en esta investigación. 
						<br><br>
						Este modelo recibe el array de puntos extraídos por el face-landmarks-detection y predice cual es el emoción resultante tomada de la captura de vídeo del sistema, además de resaltar el contorno del rostro del evaluado.
					</p>
				</v-col>
			</v-row>
		</v-container>
	</div>
</template>

<script>
// @ is an alias to /src
import Emotion from '@/components/Emotion.vue'

export default {
  name: 'Emociones',
  components: {
    Emotion
  }
}
</script>

<style scoped>
	.max {
    max-width: 1200px;
    margin: auto;
  }
</style>