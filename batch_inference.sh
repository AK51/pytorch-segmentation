#!/bin/bash
input_folder="$1"
output_folder="$2"
model="$3"
for file in "$input_folder"/*; do
    filename=$(basename -- "$file")
    input_path_file="$input_folder/$filename"
    output_path_file="$output_folder/$filename"
    echo "$input_path_file"
    echo "$output_path_file"
    python3 segnet_GY_dot.py "$input_path_file" "$output_path_file" --model="$model" --labels=classes.txt --input_blob="input_0" --output_blob="output_0" --colors=colors.txt
done

