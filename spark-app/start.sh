#!/bin/bash

/spark/bin/spark-submit \
  --packages org.postgresql:postgresql:42.2.27 \
  /app/spark-stream.py &

# Lancer une première fois le job Jira immédiatement
echo "🔄 Lancement initial du job Jira"
/spark/bin/spark-submit \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
  /app/jira_sync_spark_job.py

# Vérifier le dossier des crons
mkdir -p /etc/cron.d

# Créer le job cron (tous les jours à 00h05)
cat <<EOF > /etc/cron.d/jira_job
30 23 * * * root /spark/bin/spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 /app/jira_sync_spark_job.py >> /var/log/cron.log 2>&1
EOF

# Appliquer les droits
chmod 0644 /etc/cron.d/jira_job
crontab /etc/cron.d/jira_job

# ✅ Démarrer le démon cron
echo "Cron activé pour Jira (exécution à 23h30 chaque jour)"
crond -f -L /var/log/cron.log

