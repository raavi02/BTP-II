# BTP-II
Code for NDVI calculation given panel images and RGB-NIR

Sample command for running segment.py
```
python segment.py --base-path /home/raavi/SAM-MSCG/micasense/imageprocessing/data/000 --image-extension IMG_0125_*.tif --panel-extension IMG_0002_*.tif --use-sharp --device cuda:1 --output /home/raavi/SAM-MSCG/micasense/imageprocessing/output --checkpoint /home/raavi/SAM-MSCG/SAM-MSCG/segment-anything/ckpt/sam_vit_h_4b8939.pth --convert-to-json
```
VIT-H checkpoint can be downloaded from: [VIT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
