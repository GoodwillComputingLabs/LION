for FILE in *.darshan
do
	darshan-parser --total $FILE >> $FILE-parsed
done
