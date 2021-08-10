#!/usr/bin/env bash

RETURN=0
# Check if the files have good coding style
#
# Check the if clang-format is installed
CLANG_FORMAT=$(which clang-format)
if [ $? -ne 0 ]; then
    echo "Please install clang-format"
    exit 1
fi

DIFF=$(which colordiff)
if [ $? -ne 0 ]; then
    DIFF="$(which diff)"
fi

FILES=`git diff --cached --name-only --diff-filter=ACMR | grep -E ".\.(c|cpp|h)$"`
for FILE in $FILES; do
    echo "$FILE"
    # mkdir by using mktemp
    temp_dir=`mktemp -d` || exit 1
    nf=`git checkout-index --temp $FILE | cut -f 1` || exit 1
    base_name=`basename ${FILE}` || exit 1
    mv "${nf}" "${temp_dir}/${base_name}" || exit 1

    source_file="${temp_dir}/${base_name}"
    output_file="${temp_dir}/${base_name}.out" 

    cp .clang-format $temp_dir
    clang-format ${source_file} > ${output_file} 2>>/dev/null

    $DIFF -u -p -B --label="modified $FILE" --label="expected coding style" \
          "${source_file}" "${output_file}"
    r=$?
    if [ $r -ne 0 ]; then
        echo "Please run the command : " >&2
        echo "      clang-format -i ${FILE}" >&2
        RETURN=1
    fi
    
    rm -rf "${temp_dir}"
done

exit $RETURN
