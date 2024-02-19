@echo off

echo Run lexchatbot image with Python
winpty docker run -it --rm -h localhost -v %CD%:/app -p 8501:8501 --name lexchatbot lexchatbot
