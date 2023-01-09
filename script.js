let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");
let head = "ok";
let count = 0;

async function loadmodel(){
  const tfliteModel = await tflite.loadTFLiteModel('/keypoint_classifier.tflite');
  return tfliteModel;

}

function landmarkCalc(landmarks){
  const landmarksarray = [];
  for (let i=0; i<21 ; i++){
    landmarksarray[i] = [landmarks[i].x, landmarks[i].y]
  }
  return landmarksarray;
}

async function preprocessLandmark(landmarkarr){
  const landmark = landmarkarr;

  var base_x = 0;
  var base_y = 0;
  for (let i=0; i<21; i++){
    if (i===0){
      base_x = landmark[i][0];
      base_y = landmark[i][1];
    }
    landmark[i][0] -= base_x;
    landmark[i][1] -= base_y;
  }
  const landmark1 = [].concat.apply([],landmark);
  const max_value = Math.max.apply(null, landmark1.map(Math.abs));
  for (var i=0; i<42; i++){
    landmark1[i] /= max_value;
    }
  return landmark1;
}

async function getHandGesture(outputTensor){

  const output = await outputTensor.array();
  const array = output[0];
  let max = Math.max(...array);
  let i = array.indexOf(max);

  var gesture = "";

  if (i===0){gesture="open";}else if (i===1){gesture="close";}else if (i===2){gesture="pointer";}else if (i===3){gesture="ok";};

  return gesture;

}

async function onResultsHands(results) {
  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const landmarks = results.multiHandLandmarks[index];
      let input_tensor = tf.tensor([await preprocessLandmark(landmarkCalc(landmarks))]);
      await loadmodel().then(async function (res) {
        const gesture = await getHandGesture(res.predict(input_tensor));
         console.log(gesture,count);
        if (gesture===head){count+=1;}else{head = gesture; count=0;};
        if (count===25 && gesture=="pointer"){click_button.click(); count=0};
      }, function (err) {
        console.log(err);
    });
    }
  }
}


const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});
hands.onResults(onResultsHands);


const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({image: video});
  },
  width: 480,
  height: 480,
  frameRate: 5, // {ideal: 2, max: 5 },
});

camera_button.addEventListener('click', async function() {
  camera.start();
  // let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  // video.srcObject = stream;
});

click_button.addEventListener('click', function() {
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
});

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  const container = document.createElement('div')
  container.style.position = 'relative'
  document.body.append(container)
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  let image
  let canvas
  document.body.append('Loaded')
  click_button.addEventListener('click', async () => {
    if (image) image.remove()
    if (canvas) canvas.remove()
    let photo = document.getElementById("canvas")
    photo.toBlob( async (blob) => {
      image = await faceapi.bufferToImage(blob)
      container.append(image)
      canvas = faceapi.createCanvasFromMedia(image)
      container.append(canvas)
      const displaySize = { width: image.width, height: image.height }
      faceapi.matchDimensions(canvas, displaySize)
      const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box
        const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
        drawBox.draw(canvas);
        let name = result.toString().split(" ")[0];
        speechSynthesis.speak(new SpeechSynthesisUtterance(name));
        console.log("code last part is called");
      })
    })
  })
}



async function loadLabeledImages() {
  const labels = ['abhijeet','amay','sundari','vignesh', 'sandeep','Thor']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`./training/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
