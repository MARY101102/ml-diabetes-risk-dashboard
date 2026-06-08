# Step 9: React Frontend Dashboard

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step creates the frontend dashboard for the diabetes risk estimation system.

## Technology Used

- React
- Vite
- Tailwind CSS
- Browser Fetch API
- FastAPI backend integration

## Main Frontend Features

- Landing/header section
- Dashboard metric cards
- Diabetes risk input form
- API integration with FastAPI
- Estimated risk result card
- Calibrated probability display
- Raw probability display
- Calibration method display
- Top factors increasing estimated risk
- Top factors reducing estimated risk
- Responsible use disclaimer

## API Used

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predictions/diabetes` | Estimate diabetes-related risk |

## Responsible Use

The frontend must display the result as:

`Estimated Diabetes-Related Risk`

It must not display:

`You are diabetic`

The result is an educational machine learning estimate and not a medical diagnosis.