@echo off

echo Run lexchatbot image with Python
@REM winpty docker run -it --rm -h localhost -v %CD%:/app -p 8501:8501 --name lexchatbot lexchatbot
docker build -t lexchatbot .