for fold in 0 1 2 3 4
do
    python train.py --gpu 0 --fold ${fold} --encoder 'efficientnet-b5' --decoder 'unet' --height 384 --width 576 --blackout .1 --lovasz_hinge 0 --batch_size 16 --grad_accum 2
done
