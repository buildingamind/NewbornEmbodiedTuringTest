#!/bin/bash

# Loop through each PNG image in the source directory
for img in $HOME/Desktop/mask1_copy/*.png; do
    # Extract the filename from the full path
    filename=$(basename "$img")
    
    # Perform the cropping operation and save to the new directory
    ffmpeg -y -i "$img" -filter:v "crop=2050:1750:400:70" -frames:v 1 "$HOME/Desktop/mask1_cropped/$filename"
done