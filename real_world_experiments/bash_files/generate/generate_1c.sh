classes_sim3=("171 182 198 208 215 231 236 244 245 258")
classes_sim2=("269 279 283 294 330 340 366 370 386")
classes_sim1=("1 18 31 46 532 579 609 849 980")

sim_name="sim1"
nums=500

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for sample_cla in $classes_sim1
do
  train_cla=$sample_cla
  torchrun --nproc_per_node=8 --master_port=16663 generate_images.py --seeds=0-4999 \
      --net=training-runs/$sim_name-edm2-img256-xs-$train_cla-$nums/network-snapshot-0134217-0.050.pkl \
      --outdir=out/$sim_name-$train_cla-$nums-$sample_cla --class=$sample_cla \
      --batch=128

  torchrun --nproc_per_node=8 --master_port=16663 calculate_metrics.py calc --images=out/$sim_name-$train_cla-$nums-$sample_cla --num=5000 \
      --ref=fid-refs/$sim_name/inet-256-$sample_cla-1300.pkl --batch=64
done




