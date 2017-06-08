#! /bin/sh

if [ ! $# -eq 3 ]
then 
	echo ".sh [kona_hist] last_n_lines [kona_timeing]"
else
	awk '{if(NR==2) {print $0; next;} if(NF>0){pre=$0}if(NF==0) print pre;}' $1 > $1.processed1 
	tail -n $2 $1 > $1.processed2
	cat $1.processed1 $1.processed2 > $1.processed
	rm -rf $1.processed1 $1.processed2
	awk '{if(NR >= 2) print $0;}' $3 > $3.processed 
	pr -Jmt  $1.processed  $3.processed  > kona_combined.dat

fi
