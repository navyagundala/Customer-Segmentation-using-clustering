import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from io import BytesIO

def load_csv(file_like):
    if isinstance(file_like, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(file_like))
    else:
        df = pd.read_csv(file_like)
    return df

def preprocess(df, feature_columns=None, n_components=None, drop_na=True):
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_columns].copy()
    if drop_na:
        X = X.fillna(X.median())

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X.values)

    pca = None
    if n_components and 0 < n_components < Xs.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        Xs = pca.fit_transform(Xs)

    return Xs, feature_columns, scaler, pca

def compute_2d_pca(Xs):
    if Xs.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(Xs)
        return reduced[:,0], reduced[:,1]
    elif Xs.shape[1] == 2:
        return Xs[:,0], Xs[:,1]
    else:
        return Xs[:,0], np.zeros_like(Xs[:,0])
{
  "name": "customer-segmentation",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "axios": "^1.3.6",
    "chart.js": "^4.3.0",
    "react": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "tailwindcss": "^3.3.0"
  },
  "scripts": {
    "start": "react-scripts start"
  }
}
import React from "react";
import { Scatter } from "react-chartjs-2";
import { Chart as ChartJS, LinearScale, PointElement, Tooltip, Legend } from "chart.js";
ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

export default function ClusterViz({ sample }) {
  if (!sample || sample.length === 0) return null;

  const clusters = {};
  sample.forEach((r) => {
    if (!clusters[r.cluster]) clusters[r.cluster] = [];
    clusters[r.cluster].push({ x: r.pca_x, y: r.pca_y });
  });

  const colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"];
  const datasets = Object.keys(clusters).map((c, i) => ({
    label: `Cluster ${c}`,
    data: clusters[c],
    backgroundColor: colors[i % colors.length]
  }));

  return <Scatter data={{ datasets }} options={{ plugins: { legend: { position: "bottom" }}}} />;
}
