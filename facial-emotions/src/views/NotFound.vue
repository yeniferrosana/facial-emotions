<template>
  <div class="py-4">
    <v-container fluid class="max">
      <v-layout justify-center align-center>
        <v-flex shrink>
          <div>
            <v-img 
              src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632962216/proyecto-grado/404_b7kd9z.gif"
              lazy-src="https://res.cloudinary.com/yeniferramirez11/image/upload/v1632962216/proyecto-grado/404_b7kd9z.gif"
              height="400px"
            >
            </v-img>
          </div>
        </v-flex>
      </v-layout>
    </v-container>
  </div>
</template>

<script>

export default {
  name: 'NotFound'
}
</script>

<style scoped>
  .max {
    max-width: 1200px;
    margin: auto;
  }
</style>
