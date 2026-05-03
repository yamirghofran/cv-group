Below is the exact task breakdown needed to replicate the Roboflow basketball player identification project. The article’s system combines RF-DETR for detection, SAM2 for tracking, SigLIP + UMAP + K-means for team clustering, and SmolVLM2/ResNet for jersey number recognition. Roboflow says the full pipeline is not real-time, running about 1–2 FPS on an NVIDIA T4, mainly because SAM2 slows down when tracking many objects.  ￼

Project goal

Build a system that takes basketball game footage and outputs video where each player is:

1. detected,
2. tracked across frames,
3. assigned to a team,
4. matched with a jersey number,
5. resolved to a player name using the roster.

The article’s pipeline is:

Video → RF-DETR detections → SAM2 tracking → mask cleanup → team clustering → jersey number OCR/classification → number-player pairing → identity smoothing → roster lookup → final annotated video

⸻

1. Environment and setup tasks

Task 1.1 — Set up the repo/notebook environment

The article links an open-source Colab notebook for the project. The GitHub notebook exists in Roboflow’s public notebook repo, but it is large, around 45.3 MB, and the Colab page requires sign-in from the browser view I accessed.  ￼  ￼

You need to:

* Open the Colab notebook from the article.
* Save a copy to your own Google Drive or clone the GitHub notebook.
* Make sure the runtime has GPU enabled.
* Install dependencies used by the notebook, likely including:
    * Roboflow / Inference tools
    * RF-DETR dependencies
    * SAM2
    * Supervision
    * OpenCV
    * NumPy
    * scikit-learn
    * UMAP
    * PyTorch
    * Transformers or model-specific libraries
    * Video processing utilities

Deliverable

A working notebook or local Python environment where the first demo cells run without errors.

⸻

2. Data collection tasks

Task 2.1 — Choose basketball footage

The article works with short video clips where all players are visible in the first frame. This is important because SAM2 is prompted from first-frame detections. For longer clips, you need extra logic to re-detect and re-prompt SAM2 when players enter, leave, or tracking fails.  ￼

You need:

* 1–3 short clips, around 10–30 seconds each.
* Prefer broadcast-style basketball footage.
* First frame should show most or all players.
* Footage should have visible jersey numbers at least some of the time.
* Ideally choose one game where you can get both team rosters.

Deliverable

A folder like:

data/
  raw_videos/
    clip_001.mp4
    clip_002.mp4
  rosters/
    team_a.csv
    team_b.csv

⸻

3. Object detection with RF-DETR

The article uses RF-DETR-S as the object detector. It detects players, jersey numbers, ball, rim, referees, and other basketball-specific classes. The custom model was trained with 10 classes: ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, and rim. For player recognition, the article mainly uses the number class and the player-related classes.  ￼

Task 3.1 — Load the fine-tuned RF-DETR model

You need to either:

* use the model from the article/notebook, or
* train/fine-tune your own RF-DETR model on basketball footage.

For replication, use the article’s model first.

Task 3.2 — Run RF-DETR on sample frames

Run detection on:

* first frame of each video,
* sampled frames throughout the video,
* frames with occlusion, motion blur, and small jersey numbers.

Task 3.3 — Filter relevant detections

Keep:

* all player-related classes:
    * player
    * player-in-possession
    * player-jump-shot
    * player-layup-dunk
    * player-shot-block
* number

Optional but useful:

* ball
* rim
* referee

Task 3.4 — Save detection outputs

Save bounding boxes, class labels, confidence scores, and frame numbers.

Example output format:

{
  "frame_id": 120,
  "detections": [
    {
      "class": "player",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.91
    },
    {
      "class": "number",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.84
    }
  ]
}

Deliverable

A detection module:

src/detection.py

that can run:

python src/detection.py --video data/raw_videos/clip_001.mp4

and output detections.

⸻

4. Player tracking with SAM2

The article uses SAM2 for segmentation and tracking. The key method is: run RF-DETR on the first frame, use the player bounding boxes as prompts for SAM2, and let SAM2 track/segment each player through the clip.  ￼

Task 4.1 — Extract first-frame player boxes

From RF-DETR output, collect bounding boxes for all detected players in the first frame.

Task 4.2 — Prompt SAM2 with those boxes

Use each player bounding box as a SAM2 prompt.

Each prompt should initialize a unique player track ID:

track_id = 1 → player box 1
track_id = 2 → player box 2
...

Task 4.3 — Run SAM2 across the video

For each frame, output:

* player mask,
* track ID,
* bounding box derived from the mask,
* frame number.

Task 4.4 — Handle limitation for longer clips

The article notes that for longer clips, you need an extra component that monitors detections and re-prompts SAM2 so every player remains tracked.  ￼

For a basic replication, use short clips.

For a stronger project, implement:

* periodic RF-DETR detection every N frames,
* check whether untracked player detections exist,
* re-prompt SAM2 when new players appear,
* remove dead tracks after long absence.

Deliverable

A tracking module:

src/tracking.py

that outputs:

{
  "frame_id": 120,
  "tracks": [
    {
      "track_id": 3,
      "mask_path": "masks/frame_120_track_3.png",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

⸻

5. SAM2 mask cleanup

The article says SAM2 sometimes creates bad masks with multiple disconnected segments. For example, it may include the ball or background as part of the player. Their cleanup function keeps the main player segment and removes smaller disconnected segments that are too far away.  ￼

Task 5.1 — Find connected components in each mask

For each binary player mask:

* find connected components,
* identify the largest component,
* compute center of each component.

Task 5.2 — Remove far-away small components

Keep:

* the largest component,
* smaller components only if their center is within a distance threshold from the largest component’s center.

Remove:

* small disconnected regions far from the player body.

Task 5.3 — Validate visually

Create side-by-side videos:

* before cleanup,
* after cleanup.

Deliverable

A cleanup function:

clean_mask(mask, distance_threshold=...)

and a visual test video showing masks before/after cleanup.

⸻

6. Team clustering with SigLIP, UMAP, and K-means

The article uses an unsupervised method to separate players into teams. It extracts player crops, generates SigLIP embeddings, reduces them to 3 dimensions using UMAP, then applies K-means with 2 clusters.  ￼

Task 6.1 — Build a crop dataset

The article samples one frame per second from selected game videos, runs player detection, and crops the player region.  ￼

You need to:

* sample frames at 1 FPS,
* run RF-DETR,
* crop each detected player.

Important detail: the article does not use the full bounding box. It uses the central part of the player box to avoid noise from nearby players, crowd, or background.  ￼

Task 6.2 — Generate SigLIP embeddings

For every central player crop:

* feed crop into SigLIP,
* save embedding vector,
* save crop path and frame metadata.

Task 6.3 — Reduce embeddings with UMAP

Use UMAP to reduce high-dimensional embeddings to 3D. The article uses UMAP because it preserves local and global structure before clustering.  ￼

Task 6.4 — Cluster with K-means

Run:

KMeans(n_clusters=2)

Each cluster should correspond to one team.

Task 6.5 — Assign team labels to tracks

For each tracked player:

* collect several crops across frames,
* generate embeddings,
* classify each crop into cluster 0 or 1,
* assign the majority cluster as that player’s team.

Task 6.6 — Map cluster IDs to actual team names

Manually inspect example crops from cluster 0 and cluster 1.

Example:

cluster 0 → Lakers
cluster 1 → Warriors

Deliverable

A module:

src/team_clustering.py

that outputs:

{
  "track_id": 4,
  "team_cluster": 1,
  "team_name": "Team B"
}

⸻

7. Jersey number recognition

The article tested two approaches:

1. SmolVLM2 OCR
2. ResNet-32 classification

SmolVLM2 reached 56% accuracy out of the box, then 86% after fine-tuning on 3.6k jersey number crops. ResNet-32, trained on the same dataset reformatted for classification, reached 93% accuracy, outperforming SmolVLM2.  ￼

For replication, I would make ResNet classification your main method and SmolVLM2 your optional comparison.

⸻

Option A: SmolVLM2 OCR

Task 7A.1 — Crop jersey number detections

Use RF-DETR’s number class detections.

For every frame:

* detect jersey number boxes,
* crop the number region,
* save the crop.

Task 7A.2 — Run SmolVLM2 OCR

Prompt the model to read the jersey number.

Output should be normalized:

"0", "00", "1", ..., "99"

Task 7A.3 — Filter invalid predictions

The article notes SmolVLM2 sometimes predicts impossible values like "011" or "3000".  ￼

Reject predictions that:

* are not numeric,
* are longer than 2 digits unless your league allows it,
* are not in the known roster.

Deliverable

OCR output:

{
  "frame_id": 120,
  "number_bbox": [x1, y1, x2, y2],
  "prediction": "23",
  "confidence": null
}

⸻

Option B: ResNet jersey number classifier

Task 7B.1 — Build classification dataset

The article reformatted the jersey number crop dataset from JSONL into a standard image classification folder structure.  ￼

Example:

jersey_dataset/
  train/
    0/
    1/
    2/
    ...
    99/
  valid/
    0/
    1/
    ...
  test/
    0/
    1/
    ...

Task 7B.2 — Train or load ResNet-32

You can:

* use the notebook’s trained classifier if provided,
* train your own ResNet on jersey number crops,
* start with a smaller ResNet if easier.

Task 7B.3 — Evaluate classifier

Report:

* accuracy,
* confusion matrix,
* most confused numbers,
* performance by crop quality.

Task 7B.4 — Run classifier on number crops

For each detected jersey number crop, output:

{
  "frame_id": 120,
  "number_bbox": [x1, y1, x2, y2],
  "prediction": "23",
  "confidence": 0.88
}

Deliverable

A module:

src/number_recognition.py

with both:

predict_number_with_resnet(crop)
predict_number_with_smolvlm(crop)

or just ResNet if time is limited.

⸻

8. Pair jersey numbers to players using IoS

The article uses Intersection over Smaller Area, or IoS, to match jersey number detections to player masks. IoS is like IoU, but instead of dividing by the union area, it divides by the smaller region. Since the jersey number should be inside the player mask, IoS checks how much of the number region lies inside the player mask. They keep matches with IoS ≥ 0.9.  ￼

Task 8.1 — Convert number boxes into masks

For each RF-DETR number detection:

* create a binary mask the size of the frame,
* fill the number bounding box area.

Task 8.2 — Compute IoS with each player mask

Formula:

IoS = intersection_area / min(number_area, player_mask_area)

Since the number area is smaller, this usually becomes:

IoS = intersection_area / number_area

Task 8.3 — Assign number to player

For each number detection:

* compare with all player masks in the same frame,
* choose the player with highest IoS,
* accept only if IoS ≥ 0.9.

Deliverable

A pairing module:

src/pairing.py

that outputs:

{
  "frame_id": 120,
  "track_id": 3,
  "number_prediction": "23",
  "ios": 0.96
}

⸻

9. Resolve stable player identity

Even with good number recognition, the article does not rely on a single frame. It samples numbers every 5 frames and confirms a jersey number only once the same prediction appears three times in a row.  ￼

Task 9.1 — Store predictions per track

For each player track:

track_predictions[track_id] = ["23", "23", "8", "23", ...]

Task 9.2 — Sample every 5 frames

Only run number confirmation logic every 5 frames.

Task 9.3 — Confirm number after 3 repeated predictions

Example:

track 4 predictions: 23, 23, 23 → confirmed as #23

Task 9.4 — Add roster constraint

The article suggests accuracy can improve if predictions are constrained to numbers known to be on the court.  ￼

At minimum, constrain by team roster:

Team A valid numbers: 0, 3, 7, 11, 23...
Team B valid numbers: 1, 5, 8, 12, 30...

Reject predictions not in that team’s roster.

Deliverable

An identity resolver:

src/identity.py

that outputs:

{
  "track_id": 3,
  "team": "Team A",
  "jersey_number": "23",
  "player_name": "Player Name",
  "confirmed": true
}

⸻

10. Roster mapping

The final identity comes from combining:

team + confirmed jersey number → player name

Task 10.1 — Create roster CSVs

Example:

team,number,name
Team A,23,LeBron James
Team A,3,Anthony Davis
Team B,30,Stephen Curry

Task 10.2 — Build lookup function

def lookup_player(team, number):
    return roster[(roster.team == team) & (roster.number == number)]

Task 10.3 — Handle unknowns

If no stable prediction exists, label as:

Team A #unknown

or:

Track 4, Team A

Deliverable

A roster lookup module:

src/roster.py

⸻

11. Final video visualization

Task 11.1 — Draw outputs on each frame

For every frame, overlay:

* SAM2 player mask or bounding box,
* track ID,
* team label,
* jersey number,
* player name if confirmed.

Example label:

#23 LeBron James | Lakers | ID 4

Task 11.2 — Use different colors for teams

Example:

Team A → blue
Team B → red
Unknown → gray

Task 11.3 — Export annotated video

Use OpenCV or Supervision to write:

outputs/annotated_clip_001.mp4

Deliverable

Final demo video showing player detection, tracking, team classification, and identity recognition.

⸻

12. Evaluation tasks

You should evaluate each component separately, not only the final video.

Detection evaluation

Measure:

* player detection quality,
* number detection quality,
* false positives,
* missed numbers.

Tracking evaluation

Manually inspect:

* whether track IDs stay consistent,
* whether players swap IDs,
* whether occlusions break tracking.

Team clustering evaluation

Check:

* cluster purity,
* whether referees or crowd accidentally enter clusters,
* whether similar uniform colors confuse the model.

Number recognition evaluation

Report:

* ResNet accuracy,
* SmolVLM2 accuracy if used,
* confusion matrix,
* failure cases.

End-to-end evaluation

For each clip:

% of players correctly identified
% of frames with correct team
% of frames with correct jersey number
% of stable track IDs

⸻

Suggested split for your group of 8

Person 1 — Project lead / integration

Responsible for:

* repo structure,
* running the full notebook,
* connecting all modules,
* final demo pipeline,
* final presentation.

Main files:

main.py
config.yaml
README.md

⸻

Person 2 — RF-DETR detection

Responsible for:

* loading RF-DETR,
* running detection on frames/videos,
* filtering player and number classes,
* saving detections.

Main file:

src/detection.py

⸻

Person 3 — SAM2 tracking

Responsible for:

* prompting SAM2 with first-frame player boxes,
* generating masks and track IDs,
* exporting tracking results.

Main file:

src/tracking.py

⸻

Person 4 — Mask cleanup and tracking QA

Responsible for:

* connected-component cleanup,
* removing bad SAM2 mask fragments,
* visualizing before/after,
* tracking failure analysis.

Main file:

src/mask_cleanup.py

⸻

Person 5 — Team clustering

Responsible for:

* player crop extraction,
* SigLIP embeddings,
* UMAP reduction,
* K-means clustering,
* assigning team labels to tracks.

Main file:

src/team_clustering.py

⸻

Person 6 — Jersey number recognition

Responsible for:

* number crop extraction,
* ResNet classifier,
* optional SmolVLM2 OCR comparison,
* prediction filtering.

Main file:

src/number_recognition.py

⸻

Person 7 — Number-player pairing and identity logic

Responsible for:

* IoS matching,
* pairing number boxes to player masks,
* sampling every 5 frames,
* confirming after 3 repeated predictions,
* roster lookup.

Main files:

src/pairing.py
src/identity.py
src/roster.py

⸻

Person 8 — Evaluation, visualization, and report

Responsible for:

* annotated video generation,
* metrics,
* confusion matrix,
* failure case screenshots,
* final slides/report.

Main files:

src/visualization.py
src/evaluation.py

⸻

Minimum viable version

If you have limited time, do this version:

1. Use the article’s provided notebook/model.
2. Use one short clip where all players are visible in the first frame.
3. Run RF-DETR for player and number detection.
4. Use SAM2 for tracking.
5. Use K-means team clustering.
6. Use ResNet for jersey numbers.
7. Pair number to player with IoS ≥ 0.9.
8. Confirm identity after 3 repeated predictions.
9. Export annotated video.

This would be enough to replicate the main article result.

⸻

Strong extension ideas

Once the basic replication works, your group can improve it by adding:

* automatic re-prompting of SAM2 for long clips,
* roster-constrained number classification,
* player substitution handling,
* real-time-lite mode using a faster tracker,
* comparison between ResNet and SmolVLM2,
* ablation study:
    * full crop vs central crop,
    * with vs without UMAP,
    * IoU vs IoS,
    * one-frame prediction vs three-frame confirmation.

⸻

Final checklist

Use this as your project board:

[ ] Set up notebook/repo
[ ] Download/select basketball video clips
[ ] Create roster CSVs
[ ] Run RF-DETR on first frame
[ ] Run RF-DETR on sampled frames
[ ] Save player and number detections
[ ] Prompt SAM2 with first-frame player boxes
[ ] Track players through video
[ ] Clean SAM2 masks
[ ] Extract central player crops
[ ] Generate SigLIP embeddings
[ ] Reduce embeddings with UMAP
[ ] Cluster players with K-means
[ ] Assign tracks to teams
[ ] Extract jersey number crops
[ ] Run ResNet number classifier
[ ] Optionally run SmolVLM2 OCR
[ ] Convert number boxes to masks
[ ] Compute IoS between number boxes and player masks
[ ] Pair numbers to player track IDs
[ ] Sample predictions every 5 frames
[ ] Confirm number after 3 repeated predictions
[ ] Map team + number to roster name
[ ] Draw labels on video
[ ] Export final annotated video
[ ] Evaluate detection, tracking, clustering, and recognition
[ ] Prepare final report/presentation

The most important thing: do not treat this as one model. It is a multi-stage computer vision system, and each stage can fail independently. Your report should show the pipeline, the output of each stage, and where errors enter the system.