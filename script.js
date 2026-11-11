/*
 * ATTENTIS – AI-powered real-time focus tracker
 * ---------------------------------------------------------------
 * AI PIPELINE (runs fully in the browser, nothing is uploaded)
 *
 *  1. Webcam frame
 *  2. AI MODEL: MediaPipe Face Landmarker (pre-trained deep neural networks)
 *       - detects the face
 *       - predicts 478 3D face landmarks
 *       - predicts 52 "blendshape" scores (eyeBlink, jawOpen, ...)
 *       - predicts a 4x4 head-pose matrix
 *  3. FEATURE EXTRACTION
 *       - Eye Aspect Ratio (EAR) from eye landmarks
 *       - blink score, jaw-open score, head yaw / pitch
 *  4. PERSONAL CALIBRATION (first 3 s) learns each person's normal values
 *  5. DECISION LOGIC (hybrid AI): rules on top of the AI outputs
 *       - Drowsy      eyes closed too long, or a yawn
 *       - Distracted  head turned away too long
 *       - Focused     otherwise
 *  6. FEEDBACK: ring colour, warnings (1/3, 2/3, 3/3), sound, statistics
 */

import {FaceLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ---------- Settings (relative to each person's own baseline) ---------- */
const CAL_SECS=3, CLOSED_RATIO=0.68, BLINK_SCORE=0.55, CLOSED_SECS=1.2;
const JAW_YAWN=0.55, YAW_LIMIT=30, PITCH_LIMIT=25, AWAY_SECS=2.0, WARN_COOLDOWN=6;

const R_EYE=[33,160,158,133,153,144], L_EYE=[362,385,387,263,373,380];
const $=id=>document.getElementById(id);
const video=$('video'), canvas=$('overlay'), ctx=canvas.getContext('2d');

let landmarker, audioOn=false, actx;
let S, cal, base={ear:0.28,yaw:0,pitch:0};
resetState(); startCal();

function resetState(){
  S={closedSince:null,yawnSince:null,awaySince:null,noFaceSince:null,lastT:null,focus:0,total:0,
     blinks:0,yawns:0,warnings:0,lastWarn:0,prevState:'',frames:0,fpsT:performance.now()};
  $('blinkV').textContent='0';$('yawnV').textContent='0';$('warnV').textContent='0/3';$('focusV').textContent='–';
  $('banner').classList.remove('show');
}
function startCal(){cal={on:true,t0:null,ear:[],yaw:[],pitch:[]};}
$('resetBtn').onclick=resetState;
$('calBtn').onclick=startCal;
$('bannerClose').onclick=()=>$('banner').classList.remove('show');
$('audioBtn').onclick=()=>{
  audioOn=!audioOn; actx=actx||new AudioContext();
  $('audioBtn').textContent=audioOn?'Sound on':'Sound off'; if(audioOn)beep();
};
function beep(){
  if(!audioOn)return;
  const o=actx.createOscillator(),g=actx.createGain();
  o.frequency.value=880;g.gain.value=.15;o.connect(g);g.connect(actx.destination);
  o.start();o.stop(actx.currentTime+.25);
}

$('startBtn').onclick=async()=>{
  $('startBtn').textContent='Loading model…';
  try{
    const fs=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const opts=d=>({baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",delegate:d},
      runningMode:"VIDEO",numFaces:3,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
    try{landmarker=await FaceLandmarker.createFromOptions(fs,opts("GPU"));$('aiDev').textContent='GPU';}
    catch{landmarker=await FaceLandmarker.createFromOptions(fs,opts("CPU"));$('aiDev').textContent='CPU';}
    video.srcObject=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:640,height:480}});
    await video.play();
    // Mobile fix: wait until the video really has frames and a real size
    await new Promise(r=>{
      const ok=()=>video.readyState>=2&&video.videoWidth>0&&video.videoHeight>0;
      const t=setInterval(()=>{if(ok()){clearInterval(t);r();}},50);
    });
    canvas.width=video.videoWidth;canvas.height=video.videoHeight;
    $('idle').style.display='none';
    loop();
  }catch(e){
    $('startBtn').textContent='Start camera';
    alert('Could not start: '+e.message+'\nCamera needs HTTPS or localhost and your permission.');
  }
};

const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const eyeAR=(P,idx)=>{const p=idx.map(P);return (dist(p[1],p[5])+dist(p[2],p[4]))/(2*dist(p[0],p[3]));};
const pct=(a,q)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor((s.length-1)*q)];};
const bs=(cats,name)=>cats.find(c=>c.categoryName===name)?.score??0;
const deg=v=>Math.asin(Math.max(-1,Math.min(1,v)))*180/Math.PI;

let lastVideoTime=-1;
function loop(){
  requestAnimationFrame(loop);
  if(video.readyState<2||!video.videoWidth||!video.videoHeight)return;
  if(video.currentTime===lastVideoTime)return;
  lastVideoTime=video.currentTime;
  if(canvas.width!==video.videoWidth||canvas.height!==video.videoHeight){
    canvas.width=video.videoWidth;canvas.height=video.videoHeight; // phone rotated
  }
  const now=performance.now()/1000;
  let res;
  try{res=landmarker.detectForVideo(video,performance.now());}
  catch(e){console.warn('Skipped a frame:',e.message);return;}
  const W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);

  S.frames++;
  if(performance.now()-S.fpsT>1000){$('fps').textContent=S.frames+' frames/sec';S.frames=0;S.fpsT=performance.now();}

  if(!res.faceLandmarks.length){
    S.noFaceSince??=now;
    if(now-S.noFaceSince>1) show('idle','No face found',25,'Face the camera and check the light');
    S.lastT=null;
    return;
  }
  S.noFaceSince=null;

  let k=0,best=0; // largest face = person closest to the camera
  res.faceLandmarks.forEach((f,i)=>{const w=Math.abs(f[454].x-f[234].x);if(w>best){best=w;k=i;}});
  const L=res.faceLandmarks[k];
  const P=i=>({x:L[i].x*W,y:L[i].y*H});
  const cats=res.faceBlendshapes?.[k]?.categories||[];
  const M=res.facialTransformationMatrixes?.[k]?.data;

  const ear=(eyeAR(P,R_EYE)+eyeAR(P,L_EYE))/2;
  const blink=(bs(cats,'eyeBlinkLeft')+bs(cats,'eyeBlinkRight'))/2;
  const jaw=bs(cats,'jawOpen');
  const yaw=M?deg(M[8]):0, pitch=M?deg(M[9]):0;
  $('earV').textContent=ear.toFixed(3);
  // Live AI readout panel
  $('aiPts').textContent=L.length;
  $('aiBlink').textContent=blink.toFixed(2);
  $('aiJaw').textContent=jaw.toFixed(2);
  $('aiYaw').textContent=(yaw-base.yaw).toFixed(0)+'°';
  $('aiPitch').textContent=(pitch-base.pitch).toFixed(0)+'°';

  if(cal.on){
    cal.t0??=now;
    const p=Math.min(1,(now-cal.t0)/CAL_SECS);
    cal.ear.push(ear);cal.yaw.push(yaw);cal.pitch.push(pitch);
    show('idle','Calibrating',Math.round(p*100),'Look at the screen normally');
    if(p>=1){
      base.ear=Math.max(0.12,pct(cal.ear,.75));
      base.yaw=pct(cal.yaw,.5); base.pitch=pct(cal.pitch,.5);
      cal.on=false; S.prevState=''; S.lastT=null;
    }
    drawEyes(P);
    return;
  }

  const closed=blink>BLINK_SCORE||ear<base.ear*CLOSED_RATIO;
  if(closed){S.closedSince??=now;}
  else{
    if(S.closedSince&&now-S.closedSince<.5)S.blinks++;
    S.closedSince=null;
    base.ear=base.ear*.999+ear*.001;
  }
  $('blinkV').textContent=S.blinks;
  const closedFor=S.closedSince?now-S.closedSince:0;

  if(jaw>JAW_YAWN){S.yawnSince??=now;}
  else{ if(S.yawnSince&&now-S.yawnSince>1){S.yawns++;$('yawnV').textContent=S.yawns;} S.yawnSince=null; }
  const yawning=S.yawnSince&&now-S.yawnSince>1;

  const away=Math.abs(yaw-base.yaw)>YAW_LIMIT||Math.abs(pitch-base.pitch)>PITCH_LIMIT;
  if(away){S.awaySince??=now;}else S.awaySince=null;
  const awayFor=S.awaySince?now-S.awaySince:0;

  let state='focused';
  if(closedFor>CLOSED_SECS||yawning)state='drowsy';
  else if(awayFor>AWAY_SECS)state='distracted';

  if(state==='drowsy') show('drowsy','Drowsy',Math.min(99,Math.round(65+closedFor*12+(yawning?15:0))),'You look tired. Stay alert.');
  else if(state==='distracted') show('distracted','Distracted',Math.min(95,Math.round(70+awayFor*5)),'Bring your eyes back to the screen');
  else show('focused','Focused',Math.min(98,Math.round(70+(ear/base.ear)*25)),'Eyes open, head steady');

  // Focus time
  const dt=S.lastT?Math.min(.25,now-S.lastT):0; S.lastT=now;
  S.total+=dt; if(state==='focused')S.focus+=dt;
  if(S.total>1)$('focusV').textContent=Math.round(100*S.focus/S.total)+'%';

  if(state!=='focused'&&state!==S.prevState&&now-S.lastWarn>WARN_COOLDOWN){
    S.warnings=Math.min(3,S.warnings+1);S.lastWarn=now;
    $('warnV').textContent=S.warnings+'/3';
    $('bannerText').textContent=S.warnings>=3
      ?'Warning 3 of 3. Take a break and stretch.'
      :`Warning ${S.warnings} of 3. ${state==='drowsy'?'You look drowsy.':'Eyes back on your work.'}`;
    $('banner').classList.add('show');beep();
  }
  S.prevState=state;
  drawEyes(P);
}

function drawEyes(P){
  const c=getComputedStyle(document.body).getPropertyValue('--c').trim()||'#fff';
  [...R_EYE,...L_EYE].forEach(i=>{
    const p=P(i);ctx.beginPath();ctx.arc(p.x,p.y,2.6,0,7);
    ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=1.6;ctx.strokeStyle=c;ctx.stroke();
  });
}
function show(key,label,conf,sub){
  document.body.dataset.state=key;
  $('state').textContent=label;
  $('sub').textContent=sub;
  $('confV').textContent=conf+'%';
  $('arc').style.strokeDashoffset=295.3*(1-conf/100);
}