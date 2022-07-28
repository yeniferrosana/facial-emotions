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
					<h1 class="text-center">Seguimiento Facial</h1>
				</v-col>
				<v-col cols="12" md="6" sm="12">
					<Action/>
				</v-col>
				<v-col 
					v-if="$vuetify.breakpoint.smAndDown"
					cols="12"
					sm="12"
				>
					<p>
						En este proceso es fundamental detectar el rostro y obtener características para su posterior procesamiento y clasificación. Muchos investigadores utilizan muchos algoritmos de detección de rostros como OpenCV, DLIB, Eigenfaces, LBPH y Viola-Jones. 
						<br>
						Los algoritmos convencionales extraen los aspectos destacados del rostro, llevando la teoría de la detección de AUs de Ekman a la práctica. En este proyecto se utiliza el face-landmarks-detection para detectar esos puntos de referencias faciales.
					</p>
				</v-col>
				<!-- Version Web -->
				<v-col 
					v-if="!$vuetify.breakpoint.smAndDown"
					cols="12"
					md="6"
					sm="12"
					order="3"
					order-sm="1"
				>
					<h1>Seguimiento Facial</h1>
					<p class="text-justify mt-4 mr-4">
						En este proceso es fundamental detectar el rostro y obtener características para su posterior procesamiento y clasificación. Muchos investigadores utilizan muchos algoritmos de detección de rostros como OpenCV, DLIB, Eigenfaces, LBPH y Viola-Jones. 
						<br><br>
						Los algoritmos convencionales extraen los aspectos destacados del rostro, llevando la teoría de la detección de AUs de Ekman a la práctica. En este proyecto se utiliza el face-landmarks-detection para detectar esos puntos de referencias faciales.
					</p>
				</v-col>
			</v-row>
		</v-container>
	</div>
  <!--<div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>-->
</template>

<script>
// @ is an alias to /src
import Action from '@/components/Action.vue'

export default {
  name: 'Acciones',
  components: {
    Action
  },
}
</script>

<style scoped>
	.max {
    max-width: 1200px;
    margin: auto;
  }
</style>
