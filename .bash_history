adduser toantran
adduser khanhnguyen
adduser vdesmet
su - vdesmet
docker
uname -r
which wget
wget -qO- https://get.docker.com/ | sh
docker version
usermod -aG docker toantran
usermod -aG docker khanhnguyen
usermod -aG docker vdesmet
su - vdesmet
gpasswd -a vdesmet sudo
gpasswd -a toantran sudo
gpasswd -a khanhnguyen sudo
cd ~toantran/
su - toantran
cd /var
ls -l
cd opt
git clone https://github.com/khanhicetea/flask-skeleton.git ledis
cd ledis/
docker build -t ledis:1.0 .
cd /var/opt/ledis
git fetch --all
git reset --hard origin/master
git pull
docker build -t ledis:1.0 .
cat requirements.txt 
docker ps -a
docker images
docker images -f "dangling=true"
docker images -f "dangling=true" -q | xargs docker rmi
docker images
git pull
docker build -t ledis:1.0 .
docker run --name ledis -d -p 80:5000 ledis:1.0
docker logs -f ledis
cd ..
mkdir git-sync
cd git-sync/
git init
git remote add -f origin https://github.com/kubernetes/contrib
git config core.sparseCheckout true
echo "git-sync" >> .git/info/sparse-checkout
git pull origin master
cd git-sync/
docker build -t git-sync .
docker run -d -e "GIT_SYNC_REPO=https://github.com/khanhicetea/flask-skeleton.git" -e "GIT_SYNC_DEST=/git" -e "GIT_SYNC_BRANCH=master" -v /var/opt/ledis:/git --name=git-sync git-sync
docker ps -a
docker logs -f git-sync
cd /var/opt/ledis/
ls -l
docker ps
docker exec -it ledis bash
docker logs -f ledis
docker run --name ledis -d -p 80:5000 -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
docker stop ledis && docker rm ledis
docker run --name ledis -d -p 80:5000 -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
docker logs -f ledis 
vim Dockerfile
docker build -t ledis:1.0 .
docker images
docker run -it --rm ledis:1.0
docker run -it --rm -p 80:5000 ledis:1.0 bash
docker logs -f git-sync
docker stop git-sync
docker start git-sync
docker logs -f git-sync
git pull
docker stop git-sync && docker rm git-sync
docker run -d -e "GIT_SYNC_REPO=https://github.com/khanhicetea/flask-skeleton.git" -e "GIT_SYNC_DEST=/git" -v /var/opt/ledis:/git --name=git-sync git-sync
docker logs -f git-sync
\docker stop git-sync && docker rm git-sync
git pull
cd ../git-sync/git-sync/
mkdir -p /var/data/ledis
docker build -t ledis:1.0 .
docker run --name ledis -d -e "LEDIS_PATH=/var/data/ledis" -v /var/data/ledis:/var/data/ledis -p 80:5000 ledis:1.0
docker ps -a
docker logs -f ledis 
docker exec -it ledis bash
cd /var/opt/ledis/
ls -l
git pull
docker ps -a
docker stop ledis && docker rm ledis
docker run --name ledis -d -p 80:5000 -e "LEDIS_PATH=/var/data/ledis" -v /var/data/ledis:/var/data/ledis -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
docker exec -it ledis bash
git pull
docker stop ledis && docker rm ledis
docker run --name ledis -d -p 80:5000 -e "LEDIS_PATH=/var/data/ledis" -v /var/data/ledis:/var/data/ledis -v /var/opt/ledis/project/:/usr/src/app/project ledis:1.0
ls -l /var/data/ledis/
docker restart ledis
