data_directory=${UDA_DATA_DIR:-/home/carson/features}
python adda.py $data_directory --data=coco-m3fd --source=coco --target=m3fd --arch=resnet50 --batch-size=32 --epochs=30 --workers=8 --log=logs/adda/coco-m3fd --weighted-sample --iters-per-epoch=2500 --seed 0