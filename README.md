This code is for TMD project

Please follow this link for the installation
https://www.highvoltagecode.com/post/edge-ai-semantic-segmentation-on-nvidia-jetson
Note: if there is any problem of the cuda, please switch to CPU

There are 3 classes for labelMe.
class.txt
Use LabelMe to create the 3 segmentation of each picture
```
labelme --labels classes.txt
```
Convert to VOC format
```
python labelme2voc.py {data folder} --labels classes.txt --noviz
```
Split the data based on 10 blocks, and create 10 trial folder dir0-dir9
```
python split_custom.py --masks="{data folder}/SegmentationClass" --images="{data folder}/JPEGImages" --output="{data folder}/dir" --keep-original
```
For training:
```
python train.py {data folder}/dir0 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_0.onnx

python train.py {data folder}/dir1 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_1.onnx

python train.py {data folder}/dir2 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_2.onnx

python train.py {data folder}/dir3 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_3.onnx

python train.py {data folder}/dir4 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_4.onnx

python train.py {data folder}/dir5 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_5.onnx

python train.py {data folder}/dir6 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_6.onnx

python train.py {data folder}/dir7 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_7.onnx

python train.py {data folder}/dir8 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_8.onnx

python train.py {data folder}/dir9 --dataset=custom --epochs=100 --classes=4 --arch=fcn_resnet101 --device=cpu
python onnx_export.py --output={output folder}/fcn_reset101_9.onnx
```
After the 10 onnx files are created.
A Nvidia Nano is used for inference
Build the Nvidia segmentation from source
https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md

For Inference, details, pls reference this link, https://github.com/dusty-nv/jetson-inference/blob/master/docs/segnet-console-2.md
```
segnet_GY_dot.py {input folder} {output folder} --network={onnx file} 
```
