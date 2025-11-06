import kagglehub

# Download latest version
path = kagglehub.dataset_download("tamber/steam-video-games")

print("Path to dataset files:", path) 