"""
FastAPI application entry point.

Responsibilities:
- Initialize FastAPI app
- Load exported TorchScript model
- Define POST /predict endpoint
"""
from fastAPI import FastAPI
import torch
from os import path


app = FastAPI()


def main():
    print("Hello from workflow-minist!")


if __name__ == "__main__":
    main()
