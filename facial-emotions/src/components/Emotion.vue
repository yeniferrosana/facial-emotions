<template>
  <div>
    <!--Mostrar el reproductor de video-->
    <div class="modulo">
      <vue-element-loading 
        :active="show"
        class="loader"
        color="#30475E"
        spinner="bar-fade-scale"
        background-color="transparent" 
        size="64"
        duration="1.2"
      />
      <canvas ref="output" id="output" class="canva"></canvas>
      <video ref="webcam" id="webcam"
          style="
          visibility: hidden;
          width: 1px;
      ">
      </video>
      <h3 id="status"></h3>
    </div>
  </div>
</template>

<script>
import VueElementLoading from "vue-element-loading"

import * as tf from '@tensorflow/tfjs'
import * as FaceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'


export default {
  name: 'Emotions',
  data () {
    return {
      show: true,
      emotions: [ 'Rabia', 'Asco', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa' ],
      emotionModel: null,
      output: null,
      model: null,
      status: null,
      videoRatio: 1,
      resultWidth: 0,
      resultHeight: 0,
      points: null,
      facingMode: 'user',
      opciones: {},
      // control the UI visibilities
      isVideoStreamReady: false,
      isModelReady: false,
      initFailMessage: '',
    }
  },
  components: {
    VueElementLoading
  },
  methods: {
    /* Change Statusin H3*/
    setText( text ){
      document.getElementById( "status" ).innerText = text
    },

    /* Draw Line around the face*/
    drawLine(ctx, x1, y1, x2, y2) {
      ctx.beginPath()
      ctx.moveTo( x1, y1)
      ctx.lineTo( x2, y2)
      ctx.stroke()
    },

    /* Init Webcam*/
    async setupWebcam () {
      this.opciones = {
        audio: false, // don't capture audio
        video: {
            facingMode: 'environment' // use the rear camera if there is
        }
      };
      // if the browser supports mediaDevices.getUserMedia API
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return navigator.mediaDevices.getUserMedia(this.opciones)
          .then(stream => {
            // set <video> source as the webcam input
            let video = this.$refs.webcam
            try {
              video.srcObject = stream 
            } catch (error) {
              // support older browsers
              video.src = URL.createObjectURL(stream)
            }
            /*
              model.detect uses tf.fromPixels to create tensors.
              tf.fromPixels api will get the <video> size from the width and height attributes,
                which means <video> width and height attributes needs to be set before called model.detect
              To make the <video> responsive, I get the initial video ratio when it's loaded (onloadedmetadata)
              Then addEventListener on resize, which will adjust the size but remain the ratio
              At last, resolve the Promise.
            */
            return new Promise((resolve) => {
              // when video is loaded
              video.onloadedmetadata = () => {
                // calculate the video ratio
                this.videoRatio = video.offsetHeight / video.offsetWidth
                // add event listener on resize to reset the <video> and <canvas> sizes
                window.addEventListener('resize', this.setResultSize)
                // set the initial size
                this.setResultSize()
                this.isVideoStreamReady = true
                console.log('Webcam stream initialized')
                resolve()
              }
            })
          })
          .catch(error => {
            console.log('Failed to initialize webcam stream', error)
            throw (error)
          })
      } else {
        return Promise.reject(new Error('Your browser does not support mediaDevices.getUserMedia API'))
      }
    },
    setResultSize () {
      // get the current browser window size
      let clientWidth = document.documentElement.clientWidth
      // set max width as 450
      if (this.$vuetify.breakpoint.smAndDown) {
        this.resultWidth = Math.min(320, clientWidth)
      }
      else {
        this.resultWidth = Math.min(450, clientWidth)
      }
      // set the height according to the video ratio
      this.resultHeight = this.resultWidth * this.videoRatio
      // set <video> width and height
      /*
        Doesn't use vue binding :width and :height,
          because the initial value of resultWidth and resultHeight
          will affect the ratio got from the initWebcamStream()
      */
      this.video = this.$refs.webcam
      this.video.play()
      this.video.width = this.resultWidth
      this.video.height = this.resultHeight
      this.canvas = document.getElementById( "output" );
      this.canvas.width = this.video.width;
      this.canvas.height = this.video.height;

      this.output = this.canvas.getContext( "2d" );
      this.output.translate( this.canvas.width, 0 );
      this.output.scale( -1, 1 ); // Mirror cam
      this.output.fillStyle = "#30475E";
      this.output.strokeStyle = "#30475E";
      this.output.lineWidth = 2;
    },

    loadCustomModel () {
      this.isModelReady = false
      // load the model with loadGraphModel
      return FaceLandmarksDetection.load(
                FaceLandmarksDetection.SupportedPackages.mediapipeFacemesh)
        .then((model) => {
          this.model = model
          console.log('models loaded: ', model)
          const MODEL_URL = "./web/model/facemo.json";
          return tf.loadLayersModel(MODEL_URL)
          .then((emotionModel) => {
            this.emotionModel = emotionModel
            this.isModelReady = true
            console.log('Model loaded: ', emotionModel)
          })
        })
        .catch((error) => {
          console.log('Failed to load the models', error)
          throw (error)
        })
    },

    async predictEmotion (points) {
      let result = tf.tidy( () => {
        const xs = tf.stack( [ tf.tensor1d( points ) ] );
        return this.emotionModel.predict( xs );
      });
      let prediction = await result.data();
      result.dispose();
      // Get the index of the maximum value
      let id = prediction.indexOf( Math.max( ...prediction ) );
      return this.emotions[ id ];
    },

    /*Init Detection*/
    async detectObjects () {
    if (!this.isModelReady) return
      const video = document.querySelector( "video" );
      const faces = await this.model.estimateFaces({
        input: video,
        returnTensors: false,
        flipHorizontal: false,
      });
      this.output.drawImage(
          video,
          0, 0, video.width, video.height,
          0, 0, video.width, video.height
      );
      faces.forEach( face => {
          // Draw the bounding box
          const x1 = face.boundingBox.topLeft[ 0 ];
          const y1 = face.boundingBox.topLeft[ 1 ];
          const x2 = face.boundingBox.bottomRight[ 0 ];
          const y2 = face.boundingBox.bottomRight[ 1 ];
          const bWidth = x2 - x1;
          const bHeight = y2 - y1;
          this.drawLine( this.output, x1, y1, x2, y1 );
          this.drawLine( this.output, x2, y1, x2, y2 );
          this.drawLine( this.output, x1, y2, x2, y2 );
          this.drawLine( this.output, x1, y1, x1, y2 );

          // Add just the nose, cheeks, eyes, eyebrows & mouth
          const features = [
              "noseTip",
              "leftCheek",
              "rightCheek",
              "leftEyeLower1", "leftEyeUpper1",
              "rightEyeLower1", "rightEyeUpper1",
              "leftEyebrowLower", //"leftEyebrowUpper",
              "rightEyebrowLower", //"rightEyebrowUpper",
              "lipsLowerInner", //"lipsLowerOuter",
              "lipsUpperInner", //"lipsUpperOuter",
          ];
          this.points = [];
          features.forEach( feature => {
              face.annotations[ feature ].forEach( x => {
                  this.points.push( ( x[ 0 ] - x1 ) / bWidth );
                  this.points.push( ( x[ 1 ] - y1 ) / bHeight );
              });
          });
      });
      
      if( this.points ) {
        let emotion = await this.predictEmotion( this.points );
        this.setText( `Emoción: ${emotion}` );
      }
      else {
        this.setText( "Rostro no detectado" );
      }

      requestAnimationFrame(() => {
        this.detectObjects()
      })
    },

    loadModelAndDetection () {
      this.modelPromise = this.loadCustomModel()
      // wait for both stream and model promise finished then start detecting objects
      Promise.all([this.streamPromise, this.modelPromise])
        .then(() => {
          console.log('todo ok ')
          this.show = false // change var loader div canvas
          this.detectObjects()
        }).catch((error) => {
          console.log('Failed to init stream and/or model: ')
          this.initFailMessage = error
        })
    },
  },

  mounted () {
    this.streamPromise = this.setupWebcam()
    this.loadModelAndDetection()
  }
}
</script>

<style scoped>
.modulo {
  text-align: center;
}
.canva {
  border-radius: 1em;
}
.loader {
  z-index: 1 !important;
}

</style>
