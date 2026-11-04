#!/usr/bin/env bash
# ==============================================================================
# Provision a GKE Standard Cluster + TPU v7x (Ironwood) 2x2x4 Node Pool + GCS
# FUSE CSI Driver for Moonshot AI Kimi-K3.1-Pro (2.8T MoE) Inference
# ==============================================================================
set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-<YOUR_GCP_PROJECT_ID>}"
export REGION="${REGION:-us-central1}"
export ZONE="${ZONE:-us-central1-c}"
export CLUSTER_NAME="${CLUSTER_NAME:-tpu-v7x-inference-cluster}"
export NODE_POOL_NAME="${NODE_POOL_NAME:-tpu-v7x-16-pool}"
export GCS_BUCKET="${GCS_BUCKET:-<YOUR_GCS_MODEL_BUCKET>}"
export RESERVATION_NAME="${RESERVATION_NAME:-<YOUR_TPU_V7X_RESERVATION_NAME>}"

echo "=== 1. Creating GKE Cluster (${CLUSTER_NAME}) with Workload Identity & GCS FUSE CSI ==="
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --release-channel=rapid \
  --machine-type=e2-standard-8 \
  --num-nodes=1 \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --addons=GcsFuseCsiDriver,HttpLoadBalancing \
  --enable-ip-alias \
  --enable-gvnic

echo "=== 2. Creating Multi-Host TPU v7x (2x2x4 = 16 chips / 32 TensorCores) Node Pool ==="
gcloud container node-pools create "${NODE_POOL_NAME}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=2x2x4 \
  --num-nodes=4 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --enable-gvnic \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "=== 3. Installing LeaderWorkerSet (LWS) CRD for Multi-Host TPU Pods ==="
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/lws/releases/download/v0.5.1/manifests.yaml

echo "=== Provisioning Complete ==="
