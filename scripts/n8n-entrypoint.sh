#!/usr/bin/env sh
set -e

# Import workflows (overwrite if exists)
echo 'Importing/updating n8n workflows...'

# Import Generator workflow 
echo 'Importing Generator and Detector Starter workflow...'
n8n import:workflow --input=/workflows/generator_workflow.json || true

# Import Anomaly Processing workflow
echo 'Importing Anomaly File Watcher Pipeline workflow...'
n8n import:workflow --input=/workflows/anomaly_processing_workflow.json || true

echo 'Ensuring community nodes are available (InfluxDB)...'
mkdir -p /home/node/.n8n
if [ ! -f /home/node/.n8n/package.json ]; then
  npm init -y --prefix /home/node/.n8n >/dev/null 2>&1 || true
fi

# Determine packages to install: prefer N8N_COMMUNITY_PACKAGES env, else try common InfluxDB node names
PKGS=$(node -e 'try{const p=JSON.parse(process.env.N8N_COMMUNITY_PACKAGES||"[]");if(Array.isArray(p)&&p.length){console.log(p.join(" "))}else{console.log(["n8n-nodes-influxdb","n8n-nodes-influxdb-v2"].join(" "))}}catch(e){console.log(["n8n-nodes-influxdb","n8n-nodes-influxdb-v2"].join(" "))}')
for pkg in $PKGS; do
  echo "Attempting to install community node: $pkg"
  npm ls --prefix /home/node/.n8n "$pkg" >/dev/null 2>&1 || npm install --omit=dev --no-audit --no-fund --prefix /home/node/.n8n "$pkg" || true
done

# Hand off to official entrypoint
exec /docker-entrypoint.sh start
