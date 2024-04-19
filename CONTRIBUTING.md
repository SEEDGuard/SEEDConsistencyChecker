# :memo: Contributing to SEEDConsistencyChecker

We're excited that you're interested in contributing to SEEDConsistencyChecker! This document outlines the process for contributing to this project. Your contributions can make a real difference, and we appreciate every effort you make to help improve this project. We will be always happy to help for any problem or difficulties you may face during the contribution process.

Follow the below guidelines to start.

1. Check the available papers and the replication package provided in the README file.
2. Choose a paper that interests you, ensuring that the **"If Integrated"** field is **unchecked**.
3. Understand the paper and identify the related part which will be integrated into this component.
4. You can **Current Contributors** details, and cordinate with them for help.
5. Try running provided replication package locally to understand the core implementation.
6. Organize your code as per [What should your method contain](#what-should-your-method-contain), and follow the [Directory Structure](#directory-structure).
7. Identify the running environment, related dependencies, libraries and provided all this information by following our [Dependency Details](#dependency-details) guide.
8. Create a branch as your `name` and raise a Pull Request (PR) with your implementation on the component's repository. Refer to the [Pull Request](https://github.com/SEEDGuard/seedguard.github.io/blob/contribution_branch/CONTRIBUTING.md#repeat-submitting-pull-requests).
9. Update the Data Format table for the implemented method in the [README](https://github.com/SEEDGuard/SEEDUtils/blob/main/README.md) of the SEEDUtils Repository.
10. Once everything is done, you've successfully contributed. Great job!

## What should your method contain?

The methods should exclusively include the essential implementation, with user input data being passed, processed, and output data generated. Your method should focus solely on the testing phase of the replication package. <b>Do not</b> include any training or experimentation part inside your core implementation. Follow the [Directory Structure](#directory-structure). 

### Important Notes
Contributors are requested for the following 
1. Provide the output data files in the same format of their input data.
2. Output file should provide information on which field or key should be compared (helpful for diff checker functionality).

## Dependency Details

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

