FROM bde2020/spark-master:3.3.0-hadoop3.3

# Installer pip et requests via apk
RUN apk add --no-cache python3 py3-pip curl bash dcron \
    && pip3 install --no-cache-dir requests pymongo

 
WORKDIR /app

COPY start.sh .

# Script de démarrage : lancer le job + cron
RUN chmod +x start.sh

CMD ["./start.sh"]
