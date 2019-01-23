#wget https://github.com/kartd0094775/ML2018SPRING/releases/download/hw3/public.h5
#wget https://github.com/kartd0094775/ML2018SPRING/releases/download/hw3/private.h5
[ $3 == 'public' ] && python hw3_test.py $1 $2 'public.h5'
[ $3 == 'private' ] && python hw3_test.py $1 $2 'private.h5'