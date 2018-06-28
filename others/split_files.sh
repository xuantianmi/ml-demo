
# faces.tar.gz获取地址
# http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/
# user:adiencedb, password:adience
# 用途：https://arxiv.org/pdf/1806.02023.pdf 

# 原始的faces文件
#img_dir=faces-fake, 调试时使用的样例文件目录
img_dir=faces

# 从fold_0_data.txt文件名后缀所对应的年龄段，并查找对应的图片文件，并存到对应的年龄段目录中～
function listFilesX() {
    grep_str=$1
    dest=`echo $grep_str | sed 's/,[ ]/-/g'`
    echo 'dist:'$dest;
    echo 'grepstr:'$grep_str;

    if [ ! -d $dest  ];then
        mkdir $dest
    fi
    
    for file in `exec grep '('$grep_str')' fold_0_data.txt |awk '{print $2}'`;
    do
        echo "To find: "$file;
        for pic_dir in `ls $img_dir`;
        do
            temp="ls -l ./$img_dir/$pic_dir/|grep $file|wc -c"
            len=`ls -l ./$img_dir/$pic_dir/|grep $file|wc -c|awk '{print $1}'`;
            if [ $len == 0 ]; then
                echo 'file not found!';
            else
                echo 'find the file!'
                #ls -l ./$img_dir/$pic_dir/*$file;
                cp ./$img_dir/$pic_dir/*$file ./$dest/;
            fi
        done
    done
}

# 按照 (0 − 2),(4 − 6),(8 − 13),(15 − 20),(25 − 32),(38 − 43),(48 − 53),(60+)年龄段
# 将不同的文件存放到相应年龄段目录中～
function arrangeFiles() {
    listFilesX "0, 2"
    listFilesX "25, 32"
    listFilesX "60, 100"
}

arrangeFiles