resolution=256
nums=1000

#classes_sim1=("171 182 198 208 215 231 236 244 245 258")
#classes_sim2=("269 279 283 294 330 340 366 370 386")
#classes_sim3=("1 18 31 46 532 579 609 849 980")

sim_name=$1

if [ $sim_name == "Sim3" ]
then
  classes_sim=("171 182 198 208 215 231 236 244 245 258")
elif [ $sim_name == "Sim2" ]
then
  classes_sim=("269 279 283 294 330 340 366 370 386")
elif [ $sim_name == "Sim1" ]
then
  classes_sim=("1 18 31 46 532 579 609 849 980")
else
  echo "Wrong sim name!"
  exit 5
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3

for class_labels in $classes_sim  # select sim
do
  echo $class_labels-$nums
  python dataset_tool.py convert --source=$HOME/datas/Datasets/ImageNet/ \
      --dest=datasets/ori_datasets/$sim_name/inet-$resolution-$class_labels-$nums.zip \
      --resolution=$resolution\x$resolution --class_labels=$class_labels --nums=$nums --sim_name=$sim_name

  python dataset_tool.py encode --source=datasets/ori_datasets/$sim_name/inet-$resolution-$class_labels-$nums.zip \
      --dest=datasets/latent_datasets/$sim_name/inet-$resolution-$class_labels-$nums.zip

  python calculate_metrics.py ref \
      --data=datasets/ori_datasets/$sim_name/inet-$resolution-$class_labels-$nums.zip \
      --dest=fid-refs/$sim_name/inet-$resolution-$class_labels-$nums.pkl

done

