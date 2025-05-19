# FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04
# # Add Microsoft package repository
# RUN curl -sSL -O https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
# Add Microsoft package repository
RUN curl -sSL -O https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb

RUN sudo dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb
RUN sudo apt-get update

# Install dependencies for cv2
RUN sudo apt-get install -y libgl1-mesa-glx

# Install AzCopy
RUN sudo apt-get install azcopy
