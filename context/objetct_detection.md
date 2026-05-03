Here’s your role broken down clearly: you are responsible for object detection and player tracking. In the tutorial pipeline, your part happens before team clustering, jersey-number recognition, and roster/player identity.

Your output should give the rest of your group reliable:

frame → detected objects → tracked players → track IDs → player masks/boxes

⸻

Your section of the pipeline

The tutorial uses:

RF-DETR → detects players, numbers, referees, ball, rim, etc.
SAM2 → tracks/segments players across video
mask cleanup → removes bad SAM2 fragments
tracking output → passed to team + jersey number people

Roboflow’s tutorial says the RF-DETR detector was fine-tuned on basketball classes including ball, number, player, referee, rim, and several player action classes like player-jump-shot, player-layup-dunk, and player-shot-block.  ￼

⸻

What you need to produce

By the end, you should have four main deliverables:

1. detection.py
2. tracking.py
3. mask_cleanup.py
4. sample output files + annotated test video

Your group should be able to use your files like this:

python src/detection.py --video data/raw/clip_001.mp4
python src/tracking.py --video data/raw/clip_001.mp4 --detections outputs/detections.json
python src/visualize_tracks.py --video data/raw/clip_001.mp4 --tracks outputs/tracks.json

⸻

1. Resources you need

Core article/tutorial

Use this as your main reference:

* Roboflow article: How to Detect, Track, and Identify Basketball Players with RF-DETR, SAM2, SigLIP and ResNet.  ￼
* Roboflow notebooks repo: contains the tutorial notebooks collection.  ￼

Important note: there is a GitHub issue saying the tutorial notebook may reference the wrong model version in one place: basketball-player-detection-3-ycjdo/4 may correspond to YOLOv11s, while basketball-player-detection-3-ycjdo/13 corresponds to rfdetr-medium. Check this before presenting that you are using RF-DETR.  ￼

Detection dataset/model

Useful Roboflow Universe resource:

* basketball-player-detection-2 dataset/model
    It has 1,398 open-source player images and classes including ball, player, number, referee, rim, and action classes like player-dribble, player-jump-shot, player-layup, and player-shot-block.  ￼

The article’s tutorial uses RF-DETR-S for detection and says it was chosen for speed/accuracy balance.  ￼

Models/libraries

You need:

RF-DETR or Roboflow Inference
SAM2
Supervision
OpenCV
NumPy
PyTorch
Roboflow SDK / inference-sdk
matplotlib, tqdm, pandas

Optional but useful:

scikit-image or scipy → connected components for mask cleanup
ffmpeg → video conversion
Label Studio / CVAT / Roboflow Annotate → if you need to fix labels

⸻

2. Detection tasks

Your first job is to run object detection on the basketball video.

2.1 Input

You need from your group:

data/raw/clip_001.mp4
data/raw/clip_002.mp4

Preferably short clips:

10–30 seconds
all players visible in the first frame
not too many camera cuts
good enough resolution to see players

The “all players visible in first frame” condition matters because SAM2 tracking is initialized from first-frame detections.

⸻

2.2 Classes you should detect

For your part, separate the detections into two groups.

Main tracking classes

These are the classes you should pass to SAM2:

player
player-in-possession
player-jump-shot
player-layup-dunk
player-shot-block

Depending on the exact model/dataset version, the names might be slightly different, for example:

player-layup
player-dribble
player-screen
player-fall

Treat all player-action classes as “player-like objects.”

Other useful classes

You should still detect and save these, but you do not need to track them with SAM2 unless your group wants them:

number
ball
referee
rim
ball-in-basket

The jersey-number person will need your number detections later.

⸻

2.3 Detection output format

Save one JSON file with detections per frame.

Example:

{
  "video": "clip_001.mp4",
  "fps": 30,
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {
          "class_name": "player",
          "class_id": 3,
          "confidence": 0.91,
          "bbox_xyxy": [120, 80, 210, 310]
        },
        {
          "class_name": "number",
          "class_id": 2,
          "confidence": 0.83,
          "bbox_xyxy": [150, 130, 170, 160]
        }
      ]
    }
  ]
}

Use xyxy format:

x1, y1, x2, y2

Do not use center-width-height unless everyone agrees.

⸻

2.4 Detection quality checks

You need to inspect detections on sample frames.

Check:

[ ] Are all 10 players detected?
[ ] Are referees detected separately from players?
[ ] Are jersey numbers detected?
[ ] Does the model confuse the crowd with players?
[ ] Does the model miss players during motion blur?
[ ] Does the model detect the ball?
[ ] Are boxes tight enough for SAM2?

Create a folder:

outputs/debug_detections/
  frame_0000.jpg
  frame_0030.jpg
  frame_0060.jpg

Each image should show boxes + class labels.

⸻

3. Tracking tasks

The tutorial uses SAM2 for segmentation and tracking. RF-DETR detects the players, then SAM2 receives the player boxes as prompts and tracks those objects across the video. Roboflow describes SAM2 as a model you can prompt with bounding boxes or points, after which it tracks and segments the selected object through video frames.  ￼

⸻

3.1 Initialize SAM2 from first-frame detections

From frame 0, extract only player-like boxes:

player
player-in-possession
player-jump-shot
player-layup-dunk
player-shot-block

Then assign track IDs:

track_id 1 → first player box
track_id 2 → second player box
track_id 3 → third player box
...

Your tracking starts here.

⸻

3.2 Run SAM2 tracking

For every frame, save:

frame_id
track_id
mask
bbox from mask
confidence/score if available

Example output:

{
  "video": "clip_001.mp4",
  "tracks": [
    {
      "frame_id": 0,
      "track_id": 1,
      "bbox_xyxy": [120, 80, 210, 310],
      "mask_path": "outputs/masks/clip_001/frame_0000_track_001.png"
    },
    {
      "frame_id": 1,
      "track_id": 1,
      "bbox_xyxy": [123, 82, 213, 312],
      "mask_path": "outputs/masks/clip_001/frame_0001_track_001.png"
    }
  ]
}

⸻

3.3 Mask cleanup

SAM2 can produce masks with extra disconnected fragments. The article says this can happen when the mask accidentally includes objects like the ball or background fragments.  ￼

You should implement mask cleanup.

For each mask:

1. Find connected components.
2. Keep the largest component.
3. Optionally keep small nearby components.
4. Remove far-away small fragments.
5. Recompute the bounding box from the cleaned mask.

Function you should create:

def clean_mask(mask, distance_threshold=80, min_component_area=50):
    """
    Input: binary mask for one player.
    Output: cleaned binary mask.
    """

This part matters because the jersey-number person will later match number boxes to your player masks. If your masks include random fragments, number-player matching becomes worse.

⸻

4. “Other stuff” detection

You said you are responsible for player and other stuff detection/tracking. I would define your “other stuff” as:

number
ball
referee
rim

But do not track everything in the same way.

Player

Track with SAM2.

Output:

track_id
mask
bbox

Number

Detect with RF-DETR.

Do not track with SAM2 unless needed.

Output:

frame_id
number bbox
confidence

This goes to the jersey recognition person.

Referee

Detect with RF-DETR.

Optional:

track with ByteTrack/SAM2 if your group wants referee tracks

But for the tutorial goal, referee tracking is not essential.

Ball

Detect with RF-DETR.

Usually do not use SAM2 for ball tracking. The ball is tiny, fast, and often blurred.

Optional output:

frame_id
ball bbox
confidence

Rim

Detect with RF-DETR.

No tracking needed. Save detections if someone wants court/rim context.

⸻

5. File structure you should create

Use something like this:

project/
  data/
    raw/
      clip_001.mp4
  src/
    detection.py
    tracking.py
    mask_cleanup.py
    visualize_detections.py
    visualize_tracks.py
    utils.py
  outputs/
    detections/
      clip_001_detections.json
    tracks/
      clip_001_tracks.json
    masks/
      clip_001/
        frame_0000_track_001.png
    debug_detections/
    debug_tracks/
    videos/
      clip_001_detections.mp4
      clip_001_tracks.mp4

⸻

6. What your code should do

detection.py

Responsibilities:

[ ] Load video
[ ] Run RF-DETR / Roboflow model on frames
[ ] Save all detections
[ ] Filter player-like detections
[ ] Save debug images/video

Command:

python src/detection.py \
  --video data/raw/clip_001.mp4 \
  --output outputs/detections/clip_001_detections.json \
  --debug-video outputs/videos/clip_001_detections.mp4

⸻

tracking.py

Responsibilities:

[ ] Load video
[ ] Load detections JSON
[ ] Get first-frame player boxes
[ ] Initialize SAM2
[ ] Run SAM2 tracking
[ ] Save masks
[ ] Save track JSON

Command:

python src/tracking.py \
  --video data/raw/clip_001.mp4 \
  --detections outputs/detections/clip_001_detections.json \
  --output outputs/tracks/clip_001_tracks.json \
  --mask-dir outputs/masks/clip_001

⸻

mask_cleanup.py

Responsibilities:

[ ] Clean individual player masks
[ ] Remove disconnected mask noise
[ ] Recompute bbox from cleaned mask

⸻

visualize_tracks.py

Responsibilities:

[ ] Draw player masks or boxes
[ ] Draw track IDs
[ ] Export annotated tracking video

Example label:

ID 4

Later, the identity person can replace it with:

ID 4 | Lakers | #23

⸻

7. How your output connects to other group members

Team clustering person needs from you

They need:

player crops by track ID

You should provide a helper to crop tracked players:

outputs/crops/track_001/frame_0000.jpg
outputs/crops/track_001/frame_0030.jpg

Use the cleaned mask or bbox to crop players.

⸻

Jersey number person needs from you

They need:

number detections
player masks
track IDs

They will use number boxes + player masks to decide which number belongs to which tracked player.

So you must save:

number bbox per frame
player mask per frame
track_id per mask

⸻

Identity resolver person needs from you

They need stable track IDs.

Your main responsibility:

Track ID 3 should stay the same player as long as possible.

If IDs switch during occlusion, document it.

⸻

8. Evaluation checklist for your part

You should report these in the final presentation.

Detection metrics / checks

You probably will not have full ground truth, so use manual inspection:

Clip length: 15 seconds
Sampled frames checked: 30
Average visible players: 10
Average detected players: 9.4
Common failure: missed players during occlusion

Track:

[ ] Missed players
[ ] False crowd detections
[ ] Missed jersey numbers
[ ] Ball detection failures
[ ] Referee-player confusion

Tracking metrics / checks

Manually check:

[ ] How many track IDs were created?
[ ] Did each player keep same ID?
[ ] How many ID switches occurred?
[ ] Did SAM2 lose players after occlusion?
[ ] Did masks include wrong objects?

Create a simple table:

Clip	Players visible	Tracks created	ID switches	Main issue
clip_001	10	10	1	player overlap
clip_002	10	12	3	camera pan

⸻

9. Specific risks you should watch for

Risk 1: First frame does not show all players

SAM2 only tracks objects you initialize. If the first frame misses a player, that player will not get a track.

Solution:

Use clips where all players are visible in frame 0.

Better solution:

Run re-detection every 30 frames and initialize new SAM2 tracks for untracked players.

⸻

Risk 2: RF-DETR misses a player in the first frame

If the detector misses a player in frame 0, SAM2 will not track them.

Solution:

Manually inspect frame 0 detections.
If needed, manually add missing boxes.

For the project, manually adding a missing first-frame box is acceptable if you document it.

⸻

Risk 3: Track IDs swap during contact

Basketball has heavy occlusion. SAM2 may confuse players when bodies overlap.

Solution:

Keep clips short.
Use mask cleanup.
Record ID switches as limitations.

⸻

Risk 4: Mask fragments hurt number matching

Bad masks can cause a jersey number to be assigned to the wrong player.

Solution:

Implement connected-component cleanup.
Use cleaned masks for downstream pairing.

⸻

Risk 5: Wrong model version

As noted above, one GitHub issue says the tutorial may reference a model version that is YOLOv11s rather than RF-DETR, and suggests a different version for RF-DETR-medium. Verify the model type before claiming RF-DETR in your report.  ￼

⸻

10. Minimum version you should complete

If you are short on time, your minimum deliverable is:

[ ] Run detector on video
[ ] Save player/number/ball/referee/rim detections
[ ] Visualize detection video
[ ] Initialize SAM2 from first-frame player boxes
[ ] Track player masks through video
[ ] Save track IDs and masks
[ ] Visualize tracking video

This is enough for the rest of the group to continue.

⸻

11. Strong version you can aim for

If you want your section to stand out, add:

[ ] Mask cleanup with connected components
[ ] Manual first-frame box correction tool
[ ] Re-detection every N frames
[ ] New-track initialization for missed players
[ ] Detection confidence analysis
[ ] Tracking failure report

The manual first-frame correction tool is especially useful. It can be simple: save the first frame with boxes, let someone manually add missing boxes in a JSON file, then use those corrected boxes for SAM2.

⸻

12. Your personal task board

Setup
[ ] Open the Roboflow tutorial notebook
[ ] Confirm model version and class names
[ ] Install RF-DETR / Roboflow inference dependencies
[ ] Install SAM2 dependencies
[ ] Test GPU runtime
Detection
[ ] Load test video
[ ] Run detector on first frame
[ ] Print class names and confidence scores
[ ] Filter player-like classes
[ ] Save all detections to JSON
[ ] Save detection debug images
[ ] Export detection debug video
Tracking
[ ] Extract first-frame player boxes
[ ] Assign initial track IDs
[ ] Initialize SAM2 with boxes
[ ] Run SAM2 through full clip
[ ] Save masks per frame and track
[ ] Convert masks to bounding boxes
[ ] Save tracks to JSON
[ ] Export tracking debug video
Cleanup
[ ] Implement connected-component cleanup
[ ] Compare raw vs cleaned masks
[ ] Use cleaned masks in final track JSON
Handoff
[ ] Give team clustering person player crops
[ ] Give jersey-number person number boxes + player masks
[ ] Give identity person stable track IDs
[ ] Document failures and limitations

⸻

13. What to say in the final report

Your section can be summarized like this:

“I implemented the detection and tracking stage of the basketball player identification pipeline. I used a basketball-specific RF-DETR/Roboflow detector to detect players, jersey numbers, referees, the ball, and rim objects. I filtered player-related detections from the first frame and used them as prompts for SAM2, which generated player masks and track IDs across the clip. I then cleaned noisy SAM2 masks using connected-component filtering and exported the resulting tracks for team classification and jersey-number matching.”

That clearly explains your contribution and how it connects to the rest of the project.