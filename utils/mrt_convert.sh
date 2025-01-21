#!/bin/bash

# Use this file to create a sentence mapping documents, linking audio recordings
# to their corresponding sentence (e.g. mrt_001,went)
# The document used in this work has been adapted to exclude doubles
#
# Usage: /.mrt_convert path_to_mrt/Readme.txt path_to_hurricane/prompt-mrt.txt

reffile=$1
inputfile=$2

outsentfile="sentences.txt"
outwavfile="recordings.txt"
testfile="setencemapping.txt"

startline=`grep -n "1.	2.	3.	4.	5.	6." "$reffile"`

echo -e "$startline" | sed 's/^.*://' > "$outsentfile"
echo -e "$startline" | sed 's/^.*://' > "$outwavfile"
echo -e "$startline" | sed 's/^.*://' > "$testfile"

linenbr=${startline%%:*}
linecount=1
while IDS= read -r line; do
	words=(${line})
	nwords=${#words[@]}
	echo -en "${words[0]}">> "$outsentfile"
	echo -en "${words[0]}">> "$outwavfile"
	echo -en "${words[0]}">> "$testfile"


	for i in $(seq 1 1 $((nwords-1)))
	do
		wordtofind=`echo "${words[i]}"`
		wavfileline=`awk '{print $6}' "$inputfile" | grep -wn "$wordtofind" | sed -e 's/\(.*\):.*/\1/'`
		wavfilelinetmp=`awk '{print $6}' "$inputfile" | grep -wn ${words[i]}`

		nmatch=`echo "$wavfileline" | wc -w`
		if [ $nmatch == 0 ]
		then
			echo -en "\tNaN" >> "$outwavfile"
			echo -en "\tNaN,${words[i]}" >> "$testfile"
		elif [ $nmatch == 1 ]
		then
			jlinenbr=`echo $wavfileline | awk '{print $1}'`
			jline=`awk "NR==$jlinenbr" "$inputfile"`
			filestring=`echo "$jline" | sed -e 's/.*(\(.*\) "Now.*/\1/'`
			echo -en "\t$filestring">> "$outwavfile"
			echo -en "\t$filestring,${words[i]}" >> "$testfile"
		else
			jlinenbr=`echo $wavfileline`
			calclinenbr=`printf "%03d" $((50*(i-1)+linecount)) `
			filestring=`echo "mrt_$calclinenbr"`

			echo -en "\t$filestring">> "$outwavfile"
			echo -en "\t$filestring,${words[i]}" >> "$testfile"
		fi
		echo -en "\t${words[i]}">> "$outsentfile"
	done
	echo -en "\n" >>$outwavfile
	echo -en "\n" >>$outsentfile
	echo -en "\n" >>$testfile	
	let linecount++
done < <(tail -n "+$((linenbr+1))" "$reffile")
