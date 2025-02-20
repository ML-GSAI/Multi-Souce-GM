classes_sim3=("171 182 198 208 215 231 236 244 245 258")
classes_sim2=("269 279 283 294 330 340 366 370 386")
classes_sim1=("1 18 31 46 532 579 609 849 980")

nums=1000

# single class 1000
sim_name=sim3
for cla in $classes_sim3
do
  echo $sim_name
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=25641 train_edm2.py \
      --outdir=training-runs/$sim_name-edm2-img256-xs-$cla-$nums \
      --data=datasets/latent_datasets/$sim_name/inet-256-$cla-$nums.zip \
      --preset=edm2-img256-xs-1c-1000 \
      --batch=$(( 1024 * 8 )) --status=64Ki --snapshot=16Mi --checkpoint=64Mi
done
