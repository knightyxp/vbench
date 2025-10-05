ps -ef | grep evaluate.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep eval_video.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep eval_video.sh | grep -v grep | awk '{print $2}' | xargs kill -9