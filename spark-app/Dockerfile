FROM bde2020/spark-master:3.3.0-hadoop3.3

WORKDIR /app

# ✅ Aucun JAR Excel requis désormais
# ✅ Le seul package nécessaire est le driver PostgreSQL
CMD /spark/bin/spark-submit \
    --packages org.postgresql:postgresql:42.2.27 \
    /app/spark-stream.py
