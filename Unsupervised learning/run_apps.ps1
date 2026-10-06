$env:Path = "$PSScriptRoot\venv\Scripts;" + $env:Path

Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "api.main:app --reload --port 8000"
Start-Sleep -Seconds 3
Start-Process -NoNewWindow -FilePath "streamlit" -ArgumentList "run app/streamlit_app.py --server.port 8501"

Write-Host "Both API and Streamlit are starting..." -ForegroundColor Green
