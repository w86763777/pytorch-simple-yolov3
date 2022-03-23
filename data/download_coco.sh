# !/bin/bash
mkdir coco
cd coco

# darknet custom 5k split
wget -c "https://pjreddie.com/media/files/coco/5k.part" --header "Referer: pjreddie.com"
wget -c "https://pjreddie.com/media/files/coco/trainvalno5k.part" --header "Referer: pjreddie.com"

# official coco split
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip

# gather all images
mv train2014 all2014
mv val2014/* all2014/
rm -rf val2014

# convert darknet custom 5k split to coco format
cd ..
python generate_annotations.py
