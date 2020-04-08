
#!/bin/bash
check=false
for d in */ ; do
    #echo "$d"
    if [ $check = false ];
    then
        echo "yes"
        cd $d
        mkdir color
        mv *.jpg color/
        check=true
    else
        cd ../$d
        mkdir color
        mv *.jpg color/
    fi
done