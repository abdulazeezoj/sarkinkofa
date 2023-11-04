<div align="center">
    <h1>SARKINkofa</h1>
    <code>Vehicle & Number Plate Recognition System</code>
</div>
<hr>

## Table of Contents
- [Introduction](#introduction)
- [Technicals](#technicals)
- [Useful Commands](#useful-commands)

## Introduction
SARKINkofa is a Vehicle & Number Plate Recognition System that is built to detect and recognize vehicles and their number plates. It is built with the following technologies:
- Python
- OpenCV
- YOLOv8

SARKINkofa is a Hausa word that means "Gatekeeper". It is a project that is built to help in the security of the country by detecting and recognizing vehicles and their number plates. It is built to be used in the following places:
- Police Checkpoints
- Border Checkpoints
- Toll Gates
- Parking Lots
- Security Checkpoints
- Security Gates (e.g. Airport, Government House, Schools, etc.)

## Technicals

## HOW TOs

### How to install

#### Using Pip

To install the project using `pip`, run the following command:
```bash
pip install git+https://github.com/abdulazeezoj/sarkin-kofa.git
```

#### Using Poetry
To install the project using `poetry`, run the following command:
```bash
poetry add git+https://github.com/abdulazeezoj/sarkin-kofa.git
```

### Useful Commands

To run the project after installation, run the following command:
```bash
sarkinkofa-fs <input_folder> <output_folder>
```

To run the project in verbose mode, run the following command:
```bash
sarkinkofa-fs <input_folder> <output_folder> --verbose
```

For help, run the following command:
```bash
sarkinkofa-fs --help
```

To install completion for your shell, run the following command:
```bash
sarkinkofa-fs --install-completion
```