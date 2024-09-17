FROM mcr.microsoft.com/devcontainers/python:latest AS base

RUN sudo apt update
RUN sudo apt install -y gh

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt