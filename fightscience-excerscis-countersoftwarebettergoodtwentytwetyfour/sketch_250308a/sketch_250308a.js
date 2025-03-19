// Make sure to include p5.js and ml5.js in your HTML file:
// <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
// <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/addons/p5.dom.js"></script>
// <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>

let video;
let poseNet;
let poses = [];
let exportData = [];
let frameIdx = 0;

let legImpulseTime = 0;
let fistImpulseTime = 0;

const impulseDuration = 500; // milliseconds
const impulseForce = 20;     // additional force units
const weight = 10;           // kg
const g = 9.81;              // m/s²
const forcePerLeg = (weight * g) / 2; // force per leg

function setup() {
  createCanvas(420, 640);
  
  // Load a video file – place the file in your project folder.
  video = createVideo(['sdfgdfg.mp4'], videoLoaded);
  video.size(width, height);
  video.hide();
  
  // Initialize PoseNet
  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', function(results) {
    poses = results;
  });
}

function videoLoaded() {
  video.loop();
  video.volume(0);
}

function modelReady() {
  console.log('PoseNet model loaded');
}

function draw() {
  // Draw the current video frame
  image(video, 0, 0, width, height);
  
  if (poses.length > 0) {
    let pose = poses[0].pose;
    // Create a dictionary of keypoints for easy access
    let keypoints = {};
    for (let i = 0; i < pose.keypoints.length; i++) {
      let kp = pose.keypoints[i];
      // Convert part names to uppercase to mimic your Python keys
      keypoints[kp.part.toUpperCase()] = kp.position;
    }
    
    // Save keypoints data (with the frame index) for later export as JSON
    exportData.push({ frame: frameIdx, keypoints: keypoints });
    
    // Calculate joint angles for knees
    let leftKneeAngle = calculateAngle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE']);
    let rightKneeAngle = calculateAngle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE']);
    
    // Draw arcs at the knees to show the angles
    drawAngleArc(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE'], leftKneeAngle);
    drawAngleArc(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE'], rightKneeAngle);
    
    // Draw vector arrows from knee to hip (red) and knee to ankle (blue)
    drawArrow(keypoints['LEFT_KNEE'], p5.Vector.sub(createVector(keypoints['LEFT_HIP'].x, keypoints['LEFT_HIP'].y),
                                                   createVector(keypoints['LEFT_KNEE'].x, keypoints['LEFT_KNEE'].y)), color(255, 0, 0));
    drawArrow(keypoints['LEFT_KNEE'], p5.Vector.sub(createVector(keypoints['LEFT_ANKLE'].x, keypoints['LEFT_ANKLE'].y),
                                                   createVector(keypoints['LEFT_KNEE'].x, keypoints['LEFT_KNEE'].y)), color(0, 0, 255));
    
    drawArrow(keypoints['RIGHT_KNEE'], p5.Vector.sub(createVector(keypoints['RIGHT_HIP'].x, keypoints['RIGHT_HIP'].y),
                                                    createVector(keypoints['RIGHT_KNEE'].x, keypoints['RIGHT_KNEE'].y)), color(255, 0, 0));
    drawArrow(keypoints['RIGHT_KNEE'], p5.Vector.sub(createVector(keypoints['RIGHT_ANKLE'].x, keypoints['RIGHT_ANKLE'].y),
                                                    createVector(keypoints['RIGHT_KNEE'].x, keypoints['RIGHT_KNEE'].y)), color(0, 0, 255));
    
    // Calculate a resultant force vector for the legs
    let resultantForce = forcePerLeg * ((leftKneeAngle + rightKneeAngle) / 180);
    if (millis() - legImpulseTime < impulseDuration) {
      resultantForce += impulseForce;
    }
    // Draw force vector arrows from the ankles (using LEFT_ANKLE and RIGHT_ANKLE)
    if (keypoints['LEFT_ANKLE']) {
      drawArrow(keypoints['LEFT_ANKLE'], createVector(0, -resultantForce), color(255, 0, 0));
    }
    if (keypoints['RIGHT_ANKLE']) {
      drawArrow(keypoints['RIGHT_ANKLE'], createVector(0, -resultantForce), color(255, 0, 0));
    }
    
    // Draw hand force vectors for wrists
    let wristImpulseOffset = (millis() - fistImpulseTime < impulseDuration) ? impulseForce : 0;
    if (keypoints['LEFT_WRIST']) {
      drawArrow(keypoints['LEFT_WRIST'], createVector(30 + wristImpulseOffset, 0), color(0, 0, 255));
      drawArrow(keypoints['LEFT_WRIST'], createVector(0, -30 - wristImpulseOffset), color(0, 255, 0));
    }
    if (keypoints['RIGHT_WRIST']) {
      drawArrow(keypoints['RIGHT_WRIST'], createVector(30 + wristImpulseOffset, 0), color(0, 0, 255));
      drawArrow(keypoints['RIGHT_WRIST'], createVector(0, -30 - wristImpulseOffset), color(0, 255, 0));
    }
    
    // Kinetic linking: Draw a circle at the midpoint between hip and ankle for each side
    let leftMidX = (keypoints['LEFT_HIP'].x + keypoints['LEFT_ANKLE'].x) / 2;
    let leftMidY = (keypoints['LEFT_HIP'].y + keypoints['LEFT_ANKLE'].y) / 2;
    let rightMidX = (keypoints['RIGHT_HIP'].x + keypoints['RIGHT_ANKLE'].x) / 2;
    let rightMidY = (keypoints['RIGHT_HIP'].y + keypoints['RIGHT_ANKLE'].y) / 2;
    noStroke();
    fill(0, 255, 0);
    ellipse(leftMidX, leftMidY, 20, 20);
    ellipse(rightMidX, rightMidY, 20, 20);
    
    // Display joint angles as text
    fill(255);
    textSize(16);
    text("Left Knee: " + int(leftKneeAngle) + " deg", 20, 50);
    text("Right Knee: " + int(rightKneeAngle) + " deg", 20, 80);
    
    // (Optional) Draw all keypoints as circles
    for (let i = 0; i < pose.keypoints.length; i++) {
      let kp = pose.keypoints[i];
      noStroke();
      fill(255, 0, 0);
      ellipse(kp.position.x, kp.position.y, 10, 10);
    }
  }
  
  frameIdx++;
  
  // (Optional) Log JSON export every 100 frames to the console
  if (frameIdx % 100 === 0) {
    console.log(JSON.stringify(exportData));
  }
}

// Calculate the angle between three points (in degrees)
function calculateAngle(a, b, c) {
  let ba = createVector(a.x - b.x, a.y - b.y);
  let bc = createVector(c.x - b.x, c.y - b.y);
  return degrees(ba.angleBetween(bc));
}

// Draw an arc to represent the angle at point 'b'
function drawAngleArc(a, b, c, angle) {
  let startAngle = atan2(a.y - b.y, a.x - b.x);
  noFill();
  stroke(0, 255, 255);
  strokeWeight(2);
  arc(b.x, b.y, 60, 60, startAngle, startAngle + radians(angle));
}

// Draw an arrow from a base point in the direction of the vector 'vec'
function drawArrow(base, vec, myColor) {
  push();
  stroke(myColor);
  strokeWeight(3);
  fill(myColor);
  let end = createVector(base.x + vec.x, base.y + vec.y);
  line(base.x, base.y, end.x, end.y);
  
  // Draw arrowhead
  push();
  translate(end.x, end.y);
  let angle = atan2(vec.y, vec.x);
  rotate(angle);
  let arrowSize = 7;
  translate(-arrowSize, 0);
  triangle(0, arrowSize / 2, 0, -arrowSize / 2, arrowSize, 0);
  pop();
  pop();
}

// Capture key presses to trigger impulses
function keyPressed() {
  if (key === 'l') {
    legImpulseTime = millis();
  }
  if (key === 'f') {
    fistImpulseTime = millis();
  }
}

// (Optional) Function to download the export data as a JSON file
function downloadJSON() {
  let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData));
  let downloadAnchorNode = createA(dataStr, "export_data.json");
  downloadAnchorNode.attribute("download", "export_data.json");
  downloadAnchorNode.click();
}
