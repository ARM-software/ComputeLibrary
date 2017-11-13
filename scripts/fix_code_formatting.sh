#!/bin/bash
#FIXME: Remove this file before the release

ASTYLE_PARAMETERS=" --style=ansi \
	--indent=spaces \
	--indent-switches \
	--indent-col1-comments \
	--min-conditional-indent=0 \
	--max-instatement-indent=120 \
	--pad-oper \
	--align-pointer=name \
	--align-reference=name \
	--break-closing-brackets \
	--keep-one-line-statements \
	--max-code-length=200 \
	--mode=c \
	--lineend=linux \
	--indent-preprocessor \
	"

DIRECTORIES="./arm_compute ./src ./examples ./tests ./utils ./support"

if [ $# -eq 0 ]
then
    files=$(find $DIRECTORIES -type f \( -name \*.cpp -o -iname \*.h -o -name \*.inl -o -name \*.cl -o -name \*.cs \))
else
    files=$@
fi
for f in $files
do
    if [[ $f == *"/assembly/"* ]]
    then
        continue
    fi

    sed -i 's/\t/    /g' $f
    clang-format -i -style=file $f
    astyle -n -q $ASTYLE_PARAMETERS $f
done
