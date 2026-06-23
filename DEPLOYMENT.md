# Deploying to Hugging Face Spaces (Free)

Follow these steps to deploy your Parking Slot Detection app for free on Hugging Face Spaces.

## Step 1: Create a Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up for a free account

## Step 2: Upload Models to Hugging Face

First, upload your model files to a Hugging Face model repository:

1. Go to https://huggingface.co/new
2. Create a new **Model** repository named `parking-slot-models`
3. Upload these files to the repository:
   - `models/car_detector.pth`
   - `models/lot_detector.pt`

Or use the CLI:
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload Arvind007m/parking-slot-models models/car_detector.pth car_detector.pth
huggingface-cli upload Arvind007m/parking-slot-models models/lot_detector.pt lot_detector.pt
```

## Step 3: Create a Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `parking-slot-detection`
   - **License**: MIT (or your choice)
   - **SDK**: Docker
   - **Hardware**: CPU Basic (Free)
3. Click "Create Space"

## Step 4: Push Code to Space

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/parking-slot-detection
cd parking-slot-detection

# Copy all files from SwinDetectorSystem (except models folder and videos)
# Copy: app.py, requirements.txt, Dockerfile, templates/, *.py files

# Add and push
git add .
git commit -m "Initial deployment"
git push
```

## Step 5: Set Environment Variable (Optional)

In your Space settings, add:
- `HF_MODEL_REPO` = `Arvind007m/parking-slot-models`

## Your App Will Be Live At:
https://huggingface.co/spaces/YOUR_USERNAME/parking-slot-detection

---

## Alternative: Quick Deploy from GitHub

1. Create a new Space at https://huggingface.co/new-space
2. Select "Docker" as SDK
3. Link your GitHub repository
4. Hugging Face will auto-deploy on every push

## Notes

- **Free Tier Limits**: 2 vCPUs, 16GB RAM, no GPU
- **Cold Start**: First request may take 30-60 seconds to load models
- **Storage**: 50GB persistent storage
- Models are downloaded from Hugging Face Hub on first run
