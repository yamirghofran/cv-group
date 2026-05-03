Use the same Roboflow basketball detection dataset/model as the tutorial.

Best dataset/model for your role

Use:

basketball-player-detection-3 by Roboflow
Roboflow Universe link: search for basketball-player-detection-3-ycjdo on Roboflow Universe.

It has:

654 annotated basketball images
10 classes
pretrained model/API available
RF-DETR / YOLO model versions available

Relevant classes include:

ball
player
number
referee
rim

and player-action classes such as:

player-in-possession
player-jump-shot
player-layup-dunk
player-shot-block

This is the dataset/model most closely aligned with the tutorial, which uses RF-DETR to detect players, jersey numbers, the ball, referees, and rim objects before SAM2 tracking.  ￼

⸻

Where to find it

Go to:

Roboflow Universe → search:

basketball-player-detection-3-ycjdo

The result should be:

basketball-player-detection-3 Computer Vision Model
by Roboflow

It is listed as having 654 open-source basketball images plus a pretrained model/API.  ￼

The direct Roboflow Universe page is the one titled:

basketball-player-detection-3 Computer Vision Model

⸻

Which version should you use?

Use the version from the tutorial/notebook first.

However, there is one important warning: some discussion around the tutorial says one model version may point to a YOLO model instead of the intended RF-DETR model. So when you open the Roboflow model page, check the model version carefully.

From Roboflow’s version page, basketball-player-detection-3-ycjdo has many versions, including versions around v13–v18, with v13/v14/v16/v18 listed in 2026 and 654 images.  ￼

For your project, I’d do this:

First choice: use the exact model version used in the tutorial notebook.
Second choice: use basketball-player-detection-3-ycjdo version 13 or later.
Fallback: use basketball-player-detection-2 if version 3 gives issues.

In your report, say:

“We used the Roboflow basketball-player-detection-3 dataset/model for object detection, matching the Roboflow tutorial pipeline.”

⸻

Backup dataset

If basketball-player-detection-3 gives you issues, use:

basketball-player-detection-2 by Roboflow

It has:

1,398 open-source basketball images
pretrained model/API
classes for player, number, referee, ball, rim, and action states

Roboflow lists it as a basketball-player-detection dataset/model with 1,398 images.  ￼

This is actually larger, so it is a good fallback for detection/tracking.

Search Roboflow Universe:

basketball-player-detection-2 Roboflow

⸻

For your specific task: do you need the dataset or only the model?

For detection and tracking, you probably only need the pretrained model/API, not the raw dataset.

Use the dataset only if:

[ ] the pretrained model performs badly on your chosen video
[ ] you want to fine-tune your own detector
[ ] your teacher expects training/fine-tuning as part of the project
[ ] you need to inspect labels/classes

For following the tutorial, the simplest path is:

Use pretrained Roboflow/RF-DETR detector
→ detect players/numbers/ball/referee/rim
→ feed player boxes into SAM2
→ track players

⸻

What format should you download?

If you are fine-tuning locally:

For RF-DETR

Download/export the dataset in COCO format.

Use:

COCO JSON

because RF-DETR training commonly expects COCO-style annotations.

For YOLO fallback

Download/export in:

YOLOv8 / YOLOv11 format

if you decide to use Ultralytics YOLO instead of RF-DETR.

⸻

What your team should use each dataset for

You: detection/tracking

Use:

basketball-player-detection-3-ycjdo

for:

player detection
number detection
ball detection
referee detection
rim detection
first-frame player boxes for SAM2

Jersey-number teammate

They can start with your number detections, but may need a separate number recognition dataset/classifier later.

The detection dataset only tells you:

there is a number here

It does not necessarily classify the digit value, like 23 or 7.

Team-clustering teammate

They do not need a labeled team dataset initially. They can use your tracked player crops and do:

SigLIP embeddings → UMAP → K-means

⸻

Recommended plan

Use this exact order:

1. Open Roboflow Universe.
2. Search basketball-player-detection-3-ycjdo.
3. Click “Use this Model.”
4. Test the model on a screenshot from your chosen video.
5. Check that it detects:
   - players
   - numbers
   - ball
   - referee
   - rim
6. Use the pretrained API/model in your detection.py.
7. Only download the dataset if you need fine-tuning.
8. If model version issues happen, try basketball-player-detection-2.

⸻

My recommendation

Use:

Primary: basketball-player-detection-3-ycjdo
Backup: basketball-player-detection-2

For your role, the most important thing is not training. It is getting reliable first-frame player boxes so SAM2 can track the players. So start with the pretrained model, test it on your video, and only fine-tune if detections are bad.