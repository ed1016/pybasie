#!/bin/bash

reffile="/Users/emiliedolne/OneDrive - Imperial College London/Data/External/MRT sentences/source/Selected/Readme.txt"
inputfile=$1

outsentfile="sentences.txt"
outwavfile="recordings.txt"

startline=`grep -n "1.	2.	3.	4.	5.	6." "$reffile"`

echo -e "$startline" | sed 's/^.*://' > "$outsentfile"
echo -e "$startline" | sed 's/^.*://' > "$outwavfile"
linenbr=${startline%%:*}
linecount=1
while IDS= read -r line; do
	words=(${line})
	# echo "${words[@]}"
	nwords=${#words[@]}
	# words=`printf '%-10s' "${words[@]}"`
	# echo "${words[@]}"
	echo -en "${words[0]}">> "$outsentfile"
	echo -en "${words[0]}">> "$outwavfile"


	for i in $(seq 1 1 $((nwords-1)))
	do
		wordtofind=`echo "${words[i]}"`
		wavfileline=`awk '{print $6}' "$inputfile" | grep -wn "$wordtofind" | sed -e 's/\(.*\):.*/\1/'`
		wavfilelinetmp=`awk '{print $6}' "$inputfile" | grep -wn ${words[i]}`

		# wavfileline=`awk '{print $6}' "$inputfile" | awk -v var=$wordtofind '$0~var {print NR}'`
		# printf "$wordtofind \n"
		# echo "$wordtofind" | cat -v
		# awk '{print $6}' "$inputfile" | awk -v var=$wordtofind '$0~var {print NR,$0}'
		nmatch=`echo "$wavfileline" | wc -w`
		if [ $nmatch == 0 ]
		then
			# echo "hello"
			echo -en "\tNaN" >> "$outwavfile"
		elif [ $nmatch == 1 ]
		then
			jlinenbr=`echo $wavfileline | awk '{print $1}'`
			jline=`awk "NR==$jlinenbr" "$inputfile"`
			filestring=`echo "$jline" | sed -e 's/.*(\(.*\)"Now.*/\1/'`
			echo -en "\t$filestring">> "$outwavfile"
		else
			# echo -en "\t x">> "$outwavfile"
			jlinenbr=`echo $wavfileline`
			calclinenbr=`printf "%03d" $((50*(i-1)+linecount)) `
			filestring=`echo "mrt_$calclinenbr"`
			# awk -v var=jlinenbr "NR==var" "$inputfile" | sed "s/^.*_//" 
			# if [ $nmatch != 1 ]
			# then
			# 	for j in $(seq 2 1 $((nmatch)))
			# 	do
			# 		jlinenbr=`echo $wavfileline | awk '{print $var}' var="$j"` 
			# 		jline=`awk "NR==$jlinenbr" "$inputfile" | sed -e 's/.*(\(.*\)"Now.*/\1/'`
			# 		filestring=`echo "$filestring, $jline"`
			# 	done
			# fi
			echo -en "\t$filestring">> "$outwavfile"
		fi
		echo -en "\t${words[i]}">> "$outsentfile"
	done
	echo -en "\n" >>$outwavfile
	echo -en "\n" >>$outsentfile
	let linecount++
done < <(tail -n "+$((linenbr+1))" "$reffile")
