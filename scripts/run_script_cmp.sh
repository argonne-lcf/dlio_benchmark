#!/usr/bin/sh
#vals_0=(none gzip_0 gzip_1 gzip_2 gzip_3 gzip_4 gzip_5 gzip_6 gzip_7 gzip_8 gzip_9 lzf)
#echo "Waiting"
#while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
#do
#  sleep 2
#done
#for u in "${vals_0[@]}"
#do
#qsub ffn.sh $u
#echo "Waiting"
#while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
#do
#  sleep 2
#done
#ls /home/dhari/darshan-logs/benchmark/ffn/optimization/
#done
#
#echo "Waiting"
#while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
#do
#  sleep 2
#done
#for u in "${vals_0[@]}"
#do
#qsub cosmic.sh $u
#echo "Waiting"
#while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
#do
#  sleep 2
#done
#ls /home/dhari/darshan-logs/benchmark/cosmic/optimization/
#done

cm=(none gzip bz2 zip xz)
echo "Waiting"
while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
do
  sleep 2
done
for u in "${cm[@]}"
do
qsub candel.sh $u
echo "Waiting"
while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
do
  sleep 2
done
ls /home/dhari/darshan-logs/benchmark/cendel/optimization/
done