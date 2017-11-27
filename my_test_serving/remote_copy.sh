#!/usr/bin/env bash


echo "scp to gpu03, password=gIWUNipp23"
scp -r *.{py,sh,txt,json} root@gpu03:/data/zhaocl/test_tf_serving
