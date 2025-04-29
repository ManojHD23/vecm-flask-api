# VECM Flask API

This project uses a trained Vector Error Correction Model (VECM) to forecast future values of multivariate time series data. It includes:

- PCA for dimensionality reduction
- VECM for long-term and short-term forecasting
- Flask API for serving real-time predictions

## Usage
Send a POST request to `/predict` with JSON body containing multivariate numeric data.

## Example
```json
{
  "data": [
    [val1, val2, ..., val25]
  ]
}
