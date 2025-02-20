classes_level1=("171 182 198 208 215 231 236 244 245 258")
classes_level2=("269 279 283 294 330 340 366 370 386")
classes_level3=("1 18 31 46 532 579 609 849 980")

nums=1000
cn=10c

for level_name in level1
do
  echo $level_name
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 train_edm2.py \
      --outdir=training-runs/$level_name-edm2-img256-xs-$cn-$nums-nlr \
      --data=datasets/latent_datasets/$level_name/inet-256-$cn-$nums.zip \
      --preset=edm2-img256-xs-10c-1000-nlr \
      --batch=$(( 1024 * 6 )) --status=96Ki --snapshot=24Mi --checkpoint=96Mi
done