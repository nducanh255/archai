print_usage() {
  printf "Usage: d-> dataset\n"
}
export OPTIND=1
while getopts d:h flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        h) print_usage;
           exit 1 ;;
    esac
done
declare -A exp_descs
exp_descs['imagenet']='SimClr_ImageNet'; exp_descs['inat21']='SimClr_inat21'; exp_descs['inat21mini']='SimClr_inat21mini';
exp_descs['cifar10']='SimClr_Cifar10'; exp_descs['cifar100']='SimClr_Cifar100'; exp_descs['flower102']='SimClr_Flower102';
exp_descs['aircraft']='SimClr_Aircraft'; exp_descs['cub200']='SimClr_Cub200'; exp_descs['cars196']='SimClr_Cars196';
exp_descs['dogs120']='SimClr_Dogs120'; exp_descs['nabirds555']='SimClr_NABirds555'; exp_descs['food101']='SimClr_Food101';
exp_descs['sport8']='SimClr_Sport8'; exp_descs['mit67']='SimClr_Mit67'; exp_descs['svhn']='SimClr_Svhn';

export EXP_DESC=${exp_descs[$dataset]}
echo $EXP_DESC