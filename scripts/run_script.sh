#!/usr/bin/sh
vals_0=(0 1KB 2KB 4KB 8KB 16KB 32KB 64KB 128KB 256KB 512KB 1MB 2MB 4MB 8MB 16MB)
echo "Waiting"
while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
do
  sleep 2
done
for u in "${vals_0[@]}"
do
qsub ffn.sh $u
echo "Waiting"
while [ "$(qstat -u dhari | wc -l)" -ne 2 ]
do
  sleep 2
done
ls /home/dhari/darshan-logs/benchmark/ffn/optimization/
done
