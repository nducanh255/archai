case "$1" in
    "imagenet") export EXP_DESC='SimClr_ImageNet'
    ;;
    "inat21") export EXP_DESC='SimClr_INat21'
    ;;
    "inat21mini") export EXP_DESC='SimClr_INat21Mini'
    ;;
    "cifar10") export EXP_DESC='SimClr_Cifar10'
    ;;
    "cifar100") export EXP_DESC='SimClr_Cifar100'
    ;;
    "flower102") export EXP_DESC='SimClr_Flower102'
    ;;
    "aircraft") export EXP_DESC='SimClr_Aircraft'
    ;;
    "cub200") export EXP_DESC='SimClr_Cub200'
    ;;
    "cars196") export EXP_DESC='SimClr_Cars196'
    ;;
    "dogs120") export EXP_DESC='SimClr_Dogs120'
    ;;
    "nabirds555") export EXP_DESC='SimClr_NABirds555'
    ;;
    "food101") export EXP_DESC='SimClr_Food101'
    ;;
    "sport8") export EXP_DESC='SimClr_Sport8'
    ;;
    "mit67") export EXP_DESC='SimClr_Mit67'
    ;;
    "svhn") export EXP_DESC='SimClr_Svhn'
    ;;
esac
echo $EXP_DESC