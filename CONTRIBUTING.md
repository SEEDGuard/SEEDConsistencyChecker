# Contributing to SEEDConsistencyChecker

We're excited that you're interested in contributing to SEEDConsistencyChecker! This document outlines the process for contributing to this project. Your contributions can make a real difference, and we appreciate every effort you make to help improve this project. We will be always happy to help for any problem or difficulties you may face during the contribution process.

Follow the below guidelines to start.

## Setting up environment

Make sure you keep a track of the environment dependencies needed for the method implementation. All the library dependencies and their versions should be maintained in `requirements.txt` file.
Similar thing should be provided as DOCKER TEMPLATE inside the method, follow our [docker](https://github.com/SEEDGuard/SEEDUtils/blob/main/docker/template/Dockerfile) template.

## Directory Structure

    SEEDConsistencyChecker
    ├── core
        ├── SOME_METHOD             # Available Method for ConsistencyChecker
            ├── utils               # Additional files
            ├── __init.py__         # SOME_METHOD class or as per your configuration
            ├── requirements.txt
            ├── Dockerfile
        ├── YOUR_NEW_METHOD         # Similar to SOME_METHOD your new Method implementation
    ├── test                        # Directory for storing input and output
        ├── SOME_METHOD             # Available Method for ConsistencyChecker
            ├──  data               # provide sample data to users to run this method
            ├── testingscript.py    # perform sample testing on this data, kind of demo for users to play with.
    ├── dependencies                # Any additional common files needed by all methods
    ├── README.md
    └── main.py                     # Main file for execution
    └── otherFiles

## Additional Resources

Kindly follow our common [CONTRIBUTION](https://github.com/SEEDGuard/seedguard.github.io/blob/main/CONTRIBUTING.md) guidlines for how to create Pull Request, Issues, Commits, and additional stuffs. Below are some additional resources which can help you for your development.

More Contribution Details to be updated shortly.
