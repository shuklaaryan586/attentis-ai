# Attentis (Web) - AI-Powered Real-Time Focus Tracker

Uses your webcam and an AI face model (MediaPipe Face Landmarker) to detect whether you are
Focused, Drowsy or Distracted, and gives up to 3 warnings. All processing happens in your browser.
No video is uploaded.

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Install the "Live Server" extension (by Ritwick Dey).
3. Right-click index.html > "Open with Live Server" (or click "Go Live" at the bottom right).
4. Click "Start camera", allow the camera, and look at the screen for the 3-second calibration.

Do not double-click index.html: the camera is blocked on local files.
Internet is needed on first load to download the AI model. Use Chrome or Edge.

## Put it online (phone-friendly)
Upload this folder to Vercel or Netlify and open the link they give you.

## Files
- index.html - page structure and Live AI readout panel
- style.css  - design
- script.js  - AI pipeline and logic (commented)

## How the AI works
Webcam frame -> MediaPipe Face Landmarker (478 landmarks, blink and jaw scores, head pose)
-> features (Eye Aspect Ratio, blink score, jaw-open, yaw, pitch)
-> personal calibration (first 3 s) -> decision logic -> ring colour, warnings, sound, statistics

EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)   (Soukupova and Cech, 2016)

## Rules
- Drowsy: eyes closed more than 1.2 s, or a yawn
- Distracted: head turned or tilted away from your normal pose for more than 2 s
- Focused: otherwise
- Warnings have a 6 s cooldown

## Tuning
Constants at the top of script.js: CLOSED_RATIO, BLINK_SCORE, YAW_LIMIT, PITCH_LIMIT.
If it says Drowsy too easily, click Recalibrate with your eyes fully open.
