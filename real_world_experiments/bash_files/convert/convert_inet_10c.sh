resolution=256
cn=10c
nums=100

sim_name=$1

if [ $sim_name == "sim3" ]
then
  class_labels="171,182,198,208,215,231,236,244,245,258"
elif [ $sim_name == "sim2" ]
then
  class_labels="171,269,279,283,294,330,340,366,370,386"
elif [ $sim_name == "sim1" ]
then
  class_labels="171,1,18,31,46,532,579,609,849,980"
else
  echo "Wrong sim name!"
  exit 5
fi


echo "convert"
python dataset_tool.py convert \
    --source=$HOME/datas/Datasets/ImageNet/ \
    --dest=datasets/ori_datasets/$sim_name/inet-$resolution-$cn-$nums.zip \
    --resolution=$resolution\x$resolution --sim_name=$sim_name --class_labels=$class_labels --nums=$nums

echo "encode"
CUDA_VISIBLE_DEVICES=2,3 python dataset_tool.py encode \
    --source=datasets/ori_datasets/$sim_name/inet-$resolution-$cn-$nums.zip \
    --dest=datasets/latent_datasets/$sim_name/inet-$resolution-$cn-$nums.zip