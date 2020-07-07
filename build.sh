#!/bin/bash

case $2 in 
    container )
        ssh ajones@aj-server.local "docker build --target build -t megastep/build /home/ajones/code/megastep/docker" ;;
    code )
        docker run -it --rm --name megastepbuild -v /home/ajones/code/megastep:/code megastep/build test.sh
    * ) 
        echo "No command matching '$2'"

esac