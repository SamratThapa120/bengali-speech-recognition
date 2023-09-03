apt-get update 
apt-get install tmux -y
apt-get install ffmpeg -y
pip install --upgrade pip
pip install -r requirements.txt
cd whisper; python3 setup.py install